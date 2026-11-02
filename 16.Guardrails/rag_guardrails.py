import logging
import warnings

import nltk
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import Literal, TypedDict
from dotenv import load_dotenv
from guardrails_ai.detect_pii import DetectPII
from guardrails_ai.toxic_language import ToxicLanguage
from guardrails_ai.detect_jailbreak import DetectJailbreak
from guardrails_ai.restricttotopic import RestrictToTopic
from guardrails_ai.response_evaluator import ResponseEvaluator
from guardrails import Guard, ValidationOutcome
from guardrails.errors import ValidationError

# silence harmless warnings from Presidio's non-English recognizers and guardrails' sync fallback in Jupyter
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Could not obtain an event loop")

# load the api keys
load_dotenv()

LLM_MODEL = "openai/gpt-oss-120b"

# create the validators for each component
pii_validator = DetectPII(pii_entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="fix")
toxic_language_validator = ToxicLanguage(on_fail="fix")
jailbreak_validator = DetectJailbreak(threshold=0.8, on_fail="exception")
restrict_to_topic_validator = RestrictToTopic(valid_topics=["AI", "Machine Learning", "Data Science", "Deep Learning"],
                                              invalid_topics=["Politics", "Religion", "Sports", "Entertainment"],
                                              disable_llm=True,
                                              on_fail="exception")
response_evaluator = ResponseEvaluator(llm_callable=f"groq/{LLM_MODEL}",
                                       on_fail="noop")

input_validators = Guard().use(pii_validator,
                               toxic_language_validator,
                               restrict_to_topic_validator)

context_validators = Guard().use(jailbreak_validator)

response_validators = Guard().use(response_evaluator)

# turn off telemetry, otherwise failed exports print long tracebacks
for guard in (input_validators, context_validators, response_validators):
    guard.configure(allow_metrics_collection=False)

# build the llm
llm = ChatGroq(model=LLM_MODEL)

# graph state
class GuardState(TypedDict):
    
    original_input: str
    validated_input: str
    input_exception: bool
    
    original_context: str
    validated_context: str
    context_exception: bool
    
    original_response: str
    validated_response: str
    
    did_retrieve: bool    # dummy variable


# build the nodes
def validate_inputs(state: GuardState):
    user_input = state["original_input"]
    
    try:
        input_val_result: ValidationOutcome = input_validators.validate(user_input)
    except ValidationError as e:
        return {
            "input_exception": True,
            "validated_input": str(e)}
    else:
        return {
            "input_exception": False,
            "validated_input": input_val_result.validated_output}


def validate_context(state: GuardState):
    retrieved_context = state["original_context"]

    try:
        # check each sentence separately, otherwise the rest of the context dilutes an injected instruction
        for sentence in nltk.sent_tokenize(retrieved_context):
            context_validators.validate(sentence)
    except ValidationError as e:
        return {
            "context_exception": True,
            "validated_context": str(e)}
    else:
        return {
            "context_exception": False,
            "validated_context": retrieved_context}


def retrieve(state: GuardState):
    return {
        "did_retrieve": True
    }


def response(state: GuardState):
    validated_context = state["validated_context"]
    query = state["validated_input"]
    
    # build the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant and have knowledge about AI, Machine Learning, Data Science, and Deep Learning. Always answer questions using only the provided context."),
        ("system", "If you don't know the answer, say you don't know. Do not try to make up an answer."),
        ("system", "Here is the context: {context}"),
        ("user", "{query}")
    ])
    
    response = (prompt | llm).invoke({"context": validated_context, "query": query}).content
    
    return {"original_response": response}


def validate_response(state: GuardState):
    response = state["original_response"]
    user_query = state["validated_input"]
    
    # the evaluator expects a yes/no question, where "yes" means the response is good
    validation_question = f"Does this response answer the question: {user_query}"
    response_val_result: ValidationOutcome = response_validators.validate(response,
                                                                          metadata={"validation_question": validation_question})
    if not response_val_result.validation_passed:
        return {"validated_response": "The response is unable to answer the question based on the provided context. Please re-ask the question or provide more context."}
    else:
        return {"validated_response": response_val_result.validated_output}

# router functions
def exception_input_validation(state: GuardState) -> Literal["exception", "retrieve"]:
    if state["input_exception"]:
        return "exception"
    else:
        return "retrieve"

def exception_context_validation(state: GuardState) -> Literal["exception", "response"]:
    if state["context_exception"]:
        return "exception"
    else:
        return "response"

# build the graph
graph = StateGraph(GuardState)

# add nodes
graph.add_node("validate_inputs", validate_inputs)
graph.add_node("validate_context", validate_context)
graph.add_node("retrieve", retrieve)
graph.add_node("response", response)
graph.add_node("validate_response", validate_response)

# create edges
graph.add_edge(START, "validate_inputs")
graph.add_conditional_edges("validate_inputs", exception_input_validation, {"exception": END, "retrieve": "retrieve"})
graph.add_edge("retrieve", "validate_context")
graph.add_conditional_edges("validate_context", exception_context_validation, {"exception": END, "response": "response"})
graph.add_edge("response", "validate_response")
graph.add_edge("validate_response", END)

# compile the graph
compiled_graph = graph.compile()