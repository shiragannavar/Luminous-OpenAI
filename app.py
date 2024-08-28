import streamlit as st
import os
import pandas as pd
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI

from langchain_community.vectorstores import AstraDB
from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import OpenAIEmbeddings

import json
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document

from langchain.globals import set_llm_cache
from langchain_astradb import AstraDBCache

from langchain_openai import ChatOpenAI
from openai import OpenAI as OpenAI1

from datetime import datetime
import pytz

set_llm_cache(
    AstraDBCache(
        api_endpoint=st.secrets["ASTRA_API_ENDPOINT"],
        token=st.secrets["ASTRA_TOKEN"],
        collection_name="response_cache"
    )
)
client = OpenAI1(api_key=st.secrets['OPEN_API_KEY'])


tz_india = pytz.timezone('Asia/Kolkata')
datetime_India = datetime.now(tz_india)
# print(datetime_India)

# Define the data as a global variable (not recommended for large datasets)
data = [
    {"RangeName": "Range1", "load_lower": 0, "load_higher": 1100},
    {"RangeName": "Range2", "load_lower": 1100, "load_higher": 2200},
    {"RangeName": "Range3", "load_lower": 2200, "load_higher": 3300},
    {"RangeName": "Range4", "load_lower": 3300, "load_higher": 4400},
    {"RangeName": "Range5", "load_lower": 4400, "load_higher": 5500},
    {"RangeName": "Range6", "load_lower": 5500, "load_higher": 6600},
    {"RangeName": "Range7", "load_lower": 6600, "load_higher": 8250},
    {"RangeName": "Range8", "load_lower": 8250, "load_higher": 10450},
    {"RangeName": "Range9", "load_lower": 10450, "load_higher": 12100},
    {"RangeName": "Range10", "load_lower": 12100, "load_higher": 15400},
    {"RangeName": "Range11", "load_lower": 15400, "load_higher": 20350},
    {"RangeName": "Range12", "load_lower": 20350, "load_higher": 25300},
    {"RangeName": "Range13", "load_lower": 25300, "load_higher": 30250},
    {"RangeName": "Range14", "load_lower": 30250, "load_higher": 51150},
    {"RangeName": "Range15", "load_lower": 51150, "load_higher": 60500},
    {"RangeName": "Range16", "load_lower": 60500, "load_higher": 80300},
    {"RangeName": "Range17", "load_lower": 80300, "load_higher": 102300},
    {"RangeName": "Range18", "load_lower": 102300, "load_higher": 110000},
    {"RangeName": "Range19", "load_lower": 110000, "load_higher": 999999},
]

appliance_data = """[
  {
    "Product": "Ceiling Fan",
    "Load": 75
  },
  {
    "Product": "Table Fan",
    "Load": 50
  },
  {
    "Product": "Room Cooler",
    "Load": 250
  },
  {
    "Product": "Laptop",
    "Load": 100
  },
  {
    "Product": "Table Fan",
    "Load": 200
  },
  {
    "Product": "Room Cooler",
    "Load": 200
  },
  {
    "Product": "LED Bulb",
    "Load": 5
  },
  {
    "Product": "LED Bulb",
    "Load": 9
  },
  {
    "Product": "CFL Light",
    "Load": 15
  },
  {
    "Product": "Tubelight",
    "Load": 20
  },
  {
    "Product": "CFL Heavy",
    "Load": 30
  },
  {
    "Product": "Tubelight",
    "Load": 40
  },
  {
    "Product": "Light Bulb (Incandescent)",
    "Load": 40
  },
  {
    "Product": "Light Bulb (Incandescent)",
    "Load": 60
  },
  {
    "Product": "Light Bulb (Incandescent)",
    "Load": 100
  },
  {
    "Product": "Juicer Mixer Grinder",
    "Load": 800
  },
  {
    "Product": "Toaster",
    "Load": 800
  },
  {
    "Product": "Refrigerator (upto 200L)",
    "Load": 300
  },
  {
    "Product": "Refrigerator (upto 500L)",
    "Load": 500
  },
  {
    "Product": "Microwave Oven",
    "Load": 1400
  },
  {
    "Product": "Vacuum Cleaner",
    "Load": 1400
  },
  {
    "Product": "Washing Machine",
    "Load": 1000
  },
  {
    "Product": "Geyser/Water Heater",
    "Load": 2200
  },
  {
    "Product": "Room Heater",
    "Load": 2200
  },
  {
    "Product": "Television LED (upto 40 inch)",
    "Load": 60
  },
  {
    "Product": "Television CRT (upto 21 inch)",
    "Load": 100
  },
  {
    "Product": "Television Plasma",
    "Load": 250
  },
  {
    "Product": "Set Top Box (DTH)",
    "Load": 50
  },
  {
    "Product": "Music System",
    "Load": 300
  },
  {
    "Product": "Gaming Console",
    "Load": 200
  },
  {
    "Product": "Air Conditioner (1 Ton, 3 star)",
    "Load": 1200
  },
  {
    "Product": "Air Conditioner (1.5 Ton, 3 star)",
    "Load": 1700
  },
  {
    "Product": "Air Conditioner (2 Ton, 3 star)",
    "Load": 2300
  },
  {
    "Product": "Air Conditioner (1 Ton, Inverter)",
    "Load": 1100
  },
  {
    "Product": "Air Conditioner (1.5 Ton, Inverter)",
    "Load": 1600
  },
  {
    "Product": "Air Conditioner (2 Ton, Inverter)",
    "Load": 2100
  },
  {
    "Product": "Photo Copier",
    "Load": 2000
  },
  {
    "Product": "Office Printer/Scanner",
    "Load": 2000
  },
  {
    "Product": "Petrol Filling Machine",
    "Load": 1500
  },
  {
    "Product": "Projector",
    "Load": 600
  },
  {
    "Product": "Surveillance System",
    "Load": 100
  },
  {
    "Product": "Water Pump (0.5 HP)",
    "Load": 400
  },
  {
    "Product": "Water Pump (1 HP)",
    "Load": 800
  }
]"""


def get_range_name(target_value):
    for item in data:
        if target_value >= item["load_lower"] and target_value < item["load_higher"]:
            return item["RangeName"]
    return "Not Found"


# Only for streaming
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")


# Cache prompt for future runs
@st.cache_data()
def load_prompt():
    template = f"System: The current time is {datetime_India}, greet the customer based on the time. If the user asks for the time, politely refuse and mention that you can only assist with questions related to Luminous inverters." + """System: You are an assistant of Luminous helping customers choose the right inverter.

For each user input, detect the language accurately and respond strictly in the detected language. You are not allowed to ask the user which language they would prefer communicating in. If the user switches languages between inputs, immediately switch to the newly detected language for your response. Under no circumstances should the response be in any language other than the detected language. If the input is in Hinglish (Hindi written with English letters), respond in Hinglish without using Devanagari script, but only do so if the user's input is in Hinglish. Otherwise, respond in the default language, which is English.

Ask the customer to either list their appliances or enter the total load needed for an inverter recommendation. If the customer lists their appliances, estimate the load required based on the appliances they mention. Ask simple, direct questions to gather information about their appliances and quantities. Do not ask for the load/wattage.

When you have details of all the appliances the user listed, ask if there are more appliances the user wishes to use with the inverter. Always ask relevant questions about the appliances like AC ton, size of TV, type of light, type of fans, water pump power, or fridge capacity if they are listed in the appliance_data JSON. Use the load data only from the APPLIANCE DATA, and if the appliance is not present, assume a value based on your knowledge. Do not show the assumptions.  Do not Explain the total load calculation.

If the human says they do not have more appliances to add, ask the user what type of inverter they are looking for (GTI, Solar or Hybrid) and then ask the user to type in their average running load in percentage, return the total load of all the appliances the user mentioned in JSON format, with the key 'wattage', average running load in integer format ranging from 1 to 100 based on user input as 'per_load' and the type of inverter as 'type' which should be wither GTI, Solar ot Hybrid value and the user detected language based on user input as 'detected_lanuage' . Multiply the sum of all the loads by the quality of appliances entered by the human. If the user has entered or given the total load instead of the appliances, return the total load itself as 'wattage' and don't ask for appliances. Do not return any other key other than wattage, type and detected language.

If the user engages in small talk, respond accordingly. If the question is not related to inverter recommendation, deny responding and tell the human to ask questions only related to Luminous inverters. If user asks for any inverter suggestions that are not made by Luminous, STRICTLY DO NOT suggest anything to the user. If user asks for inverters that exceed the range, then politely apologise to the user and ask him to contact customer support. *STRICTLY DO NOT SUGGEST OTHER COMPANY PRODUCTS*. If a user lists a single appliance that is beyond the watt capacity of Luminous inverters, apologise to user and ask them to contact customer service. DO NOT SUGGEST OTHER COMPANY PRODUCTS UNDER ANY CIRCUMSTANCES. *YOU ARE ONLY ALLOWED TO SUGGEST THINGS RELATED TO LUMINOUS AND LUMINOUS PRODUCTS ONLY*.

Respond only using the history and context. Respond in natural language without showing any piece of code. No code should ever be seen by the user. If a user enters their list of appliances, show the watt of each appliance and the final sum only. Do not show the calculation process (This step should be followed for all languages).

**Note: The final answer should STRICTLY be in the user-detected language and ONLY that language. No other language should be used in the response.**

HISTORY: {history}
QUESTION: {user_question}
APPLIANCE DATA: {appliance_data}
YOUR ANSWER (strictly in the detected language or default to English):"""

    return ChatPromptTemplate.from_messages([("system", template)])


prompt = load_prompt()


@st.cache_data()
def load_prompt1():
    template = """You are an AI assistant providing the best inverter recommendation based on the input.

    CONTEXT: {context} QUESTION: {user_question} YOUR ANSWER:"""
    return ChatPromptTemplate.from_messages([("system", template)])


prompt_new = load_prompt1()


# Cache Azure OpenAI Chat Model for future runs
@st.cache_resource()
def load_chat_model():
    return ChatOpenAI(
        temperature=0.3,
        model='gpt-4o',
        streaming=True,
        verbose=True
    )

chat_model = load_chat_model()



@st.cache_resource()
def load_chat_model_nostream():
    return ChatOpenAI(
        temperature=0,
        model='gpt-4o',
        verbose=True
    )

chat_model_nostream = load_chat_model_nostream()


# Cache the Astra DB Vector Store for future runs
@st.cache_resource(show_spinner='Connecting to Astra')
def load_retriever():
    # Connect to the Vector Store
    vector_store = AstraDB(
        embedding=OpenAIEmbeddings(),
        collection_name="luminous_calculator_3",
        api_endpoint=st.secrets["ASTRA_API_ENDPOINT"],
        token=st.secrets["ASTRA_TOKEN"]
    )

    # Get the retriever for the Chat Model
    metadata_field_info = [
        AttributeInfo(
            name="range",
            description="The name of the range in which the load is",
            type="string",
        ),
        AttributeInfo(
            name="load_lower_range",
            description="the lower limit of load requirement",
            type="integer",
        ),
        AttributeInfo(
            name="load_higher_range",
            description="The upper limit of load requirement",
            type="integer",
        ),
        AttributeInfo(
            name="modelcategory1",
            description="This represents the category of the inverter model which is GTI",
            type="string",
        ),
        AttributeInfo(
            name="modelcategory2",
            description="This represents the category of the inverter model which is GTI",
            type="string",
        ),
        AttributeInfo(
            name="modelcategory3",
            description="This represents the category of the inverter model which is Solar Inverter",
            type="string",
        ),
        AttributeInfo(
            name="modelcategory4",
            description="This represents the category of the inverter model which is Hybrid Inverter",
            type="string",
        ),
    ]

    document_content_description = "Inverter details to be recommended to the human where the human input should be between load_lower_range and load_higher_range"
    llm = OpenAI(temperature=0, openai_api_key=st.secrets['OPEN_API_KEY'])

    retriever = SelfQueryRetriever.from_llm(
        llm, vector_store, document_content_description, metadata_field_info, verbose=True
    )
    return retriever


retriever = load_retriever()

#Sidebar
st.sidebar.image("https://i0.wp.com/opensource.org/wp-content/uploads/2023/01/datastax-logo-square_transparent-background.png", caption="Using DataStax AstraDB and OpenAI")

# Start with empty messages, stored in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Draw a title and some markdown
st.title("Luminous Inverters")
st.markdown("""POC for recommending the right product""")

# Draw all messages, both user and bot so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Draw the chat input box
if user_question := st.chat_input("How may I help you?"):
    st.session_state.messages.append({"role": "human", "content": user_question})

    # Draw the user's question
    with st.chat_message('human'):
        st.markdown(user_question)

    # UI placeholder to start filling with agent response
    with st.chat_message('assistant'):
        response_placeholder = st.empty()

    # Generate the answer by calling Azure OpenAI's Chat Model
    inputs = RunnableMap({
        'user_question': lambda x: x['user_question'],
        'history': lambda x: x['history'],
        'appliance_data': lambda x: x['appliance_data']
    })

    inputs_new = RunnableMap({
        'context': lambda x: retriever.get_relevant_documents(x['user_question']),
        'user_question': lambda x: x['user_question'],
        'history': lambda x: x['history']
    })

    chain = inputs | prompt | chat_model_nostream
    response = chain.invoke(
        {'appliance_data': appliance_data, 'user_question': user_question, 'history': st.session_state.messages},
        config={'callbacks': [StreamHandler(response_placeholder)]})

    try:
        json_string = response.content[response.content.find("{") + 1: response.content.rfind("}")]
        data_json = "{" + json_string + "}"
        a = json.loads(data_json)
        print(a)
        # print(a["type"])
        # print(a["per_load"])
        # print("Valid json")
        if (a["wattage"] >=0):
            new_load = int(a['wattage']) * (int(a['per_load'])/100)

            range_name = get_range_name(new_load)
            print(range_name)

            chain_n = inputs_new | prompt_new | chat_model
            response = chain_n.invoke({'user_question': f'tell me details of the inverters where range is where the range is {range_name} in {a['detected_language']} language where the category is {a['type']}. Do not show the range name',
                                          'history': st.session_state.messages},
                                      config={'callbacks': [StreamHandler(response_placeholder)]})
            answer = response.content
            st.session_state.messages.append({"role": "ai", "content": answer})
            response_placeholder.markdown(answer)
    except ValueError as e:
        answer = response.content
        st.session_state.messages.append({"role": "ai", "content": answer})
        response_placeholder.markdown(answer)