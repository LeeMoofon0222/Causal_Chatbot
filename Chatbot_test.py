import streamlit as st
import requests
from typing import Optional
import json
import os
from dotenv import load_dotenv
import sqlite3


def run_flow(message: str,
             endpoint: str,
             output_type: str = "chat",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             api_key: Optional[str] = None,
             session_id: Optional[str] = None) -> dict:
    
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
        "session_id": session_id,
    }
    headers = None
    if tweaks:
        payload["tweaks"] = tweaks
    if api_key:
        headers = {"x-api-key": api_key}
    
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()


def format_message(message: str):
    lines = message.split('\n')
    formatted_message = ""
    for line in lines:
        formatted_message += line.strip() + "\n"
    return formatted_message


def remove_json_object(json_data):
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    if not isinstance(json_data, dict) or 'edges' not in json_data:
        st.error("Project does not exsist")
        return None

    edges = json_data['edges']
    updated_edges = []

    for edge in edges:
        if isinstance(edge, dict):
            updated_edge = {
                "x": edge.get('x', ''),
                "y": edge.get('y', ''),
                "imp": round(edge.get('imp', 0), 2),
                "co": round(edge.get('co', 0), 2) if edge.get('co') is not None else None
            }
            if updated_edge['imp'] != 0:
                updated_edges.append(updated_edge)

    return updated_edges


def try_upload_json(file_content,email,project_id):
    save_dir = f"json_file_storage/{email}"
    file_content = remove_json_object(file_content)  # filter out json objects

    if file_content is None:
        if os.path.exists(current_dir + "/json_file_storage" + f"/{email}/{project_id}.json"):
            os.remove(current_dir + "/json_file_storage" + f"/{email}/{project_id}.json")
        return None
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    saved_file_path = os.path.join(save_dir, f"{project_id}.json")

    with open(saved_file_path, 'w') as file:
        json.dump(file_content, file, indent=2, ensure_ascii=False)


def model_select(model):
    if model == "Llama3 8b":
        model_name = "llama3:8b"
        model_type = "llama"
    elif model == "GPT 4o":
        model_name = "gpt-4o"
        model_type = "openai"
    elif model == "GPT 4o mini":
        model_name = "gpt-4o-mini"
        model_type = "openai"
    return model_name, model_type


def delete_session(session_id):

    url = f"http://127.0.0.1:7860/api/v1/monitor/messages/session/{session_id}"

    response = requests.delete(url)

    if response.status_code != 204:
        print(f"Failed to clear session messages: {response.status_code} - {response.text}")


def request_login(email, password):
    url = "http://192.168.50.3:25000/api/v1/user/login"
    headers = {"Content-Type": "application/json"}
    payload = {
        "email": email,
        "password": password
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        token = response.json()
        return token['access_token']
    except:
        st.error("Wrong email or password")
        return None


def request_json(email, password, project_id):
    url = f"http://192.168.50.3:25000/api/v1/dataset_groups/{project_id}/causal_graph/edges"
    access_token = request_login(email, password)
    if access_token is None:
        return None
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers)
    content = response.json()
    try_upload_json(content,email,project_id)
    

def load_session(session_id):
    try:
        from langflow.memory import get_messages

        # Define your filters
        sender = "User"
        sender_name = "User"
        session_id = session_id
        order_by = "timestamp"
        order = "ASC"
        limit = 10

        # Retrieve messages
        user_messages = get_messages(
            sender=sender,
            sender_name=sender_name,
            session_id=session_id,
            order_by=order_by,
            order=order,
            limit=limit
        )

        # Define your filters
        sender = "Machine"
        sender_name = "AI"
        session_id = session_id
        order_by = "timestamp"
        order = "ASC"
        limit = 10

        # Retrieve messages
        AI_messages = get_messages(
            sender=sender,
            sender_name=sender_name,
            session_id=session_id,
            order_by=order_by,
            order=order,
            limit=limit
        )

        st.session_state.messages = []

        for i in range(len(AI_messages)):
            st.session_state.messages.append({
                "role": "user",
                "content": user_messages[i].text,
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": AI_messages[i].text,
            })
    except Exception as e:
        return f"Fail to load memory or it's empty"


def txt_to_string(uploaded_file):
    try:
        if uploaded_file is not None:
            content = uploaded_file.read().decode('utf-8')
            return content
        else:
            return ""
    except Exception as e:
        return f"An error occurred: {e}"


def sql_create():
    conn = sqlite3.connect('messages.db')

    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS message (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        flow_id TEXT,
        sender TEXT,
        sender_name TEXT,
        session_id TEXT,
        text TEXT,
        files TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()


def model_tweaks(model_type, model_name):
    TWEAKS = {
        "Chroma-AD9ep": {
            "allow_duplicates": False,
            "chroma_server_cors_allow_origins": "",
            "chroma_server_grpc_port": None,
            "chroma_server_host": "",
            "chroma_server_http_port": None,
            "chroma_server_ssl_enabled": False,
            "collection_name": "langflow",
            "limit": None,
            "number_of_results": 5,
            "persist_directory": "",
            "search_query": "",
            "search_type": "Similarity"
        },
        "ChatInput-baxPA": {
            "files": "",
            "sender": "User",
            "sender_name": "User",
            "session_id": sessionID
        },
        "ChatOutput-8U3qC": {
            "data_template": "{text}",
            "input_value": "",
            "sender": "Machine",
            "sender_name": "AI",
            "session_id": sessionID
        },
        "ParseData-6hQ9b": {
            "sep": "\n",
            "template": "{text}"
        },
        "GroupNode-cDFw7": {
            "path_File-K3ohz": current_dir + "/json_file_storage" + f"/{email}/{project_id}.json",
            "code_File-K3ohz": "from pathlib import Path\n\nfrom langflow.base.data.utils import TEXT_FILE_TYPES, parse_text_file_to_data\nfrom langflow.custom import Component\nfrom langflow.io import BoolInput, FileInput, Output\nfrom langflow.schema import Data\n\n\nclass FileComponent(Component):\n    display_name = \"File\"\n    description = \"A generic file loader.\"\n    icon = \"file-text\"\n\n    inputs = [\n        FileInput(\n            name=\"path\",\n            display_name=\"Path\",\n            file_types=TEXT_FILE_TYPES,\n            info=f\"Supported file types: {', '.join(TEXT_FILE_TYPES)}\",\n        ),\n        BoolInput(\n            name=\"silent_errors\",\n            display_name=\"Silent Errors\",\n            advanced=True,\n            info=\"If True, errors will not raise an exception.\",\n        ),\n    ]\n\n    outputs = [\n        Output(display_name=\"Data\", name=\"data\", method=\"load_file\"),\n    ]\n\n    def load_file(self) -> Data:\n        if not self.path:\n            raise ValueError(\"Please, upload a file to use this component.\")\n        resolved_path = self.resolve_path(self.path)\n        silent_errors = self.silent_errors\n\n        extension = Path(resolved_path).suffix[1:].lower()\n\n        if extension == \"doc\":\n            raise ValueError(\"doc files are not supported. Please save as .docx\")\n        if extension not in TEXT_FILE_TYPES:\n            raise ValueError(f\"Unsupported file type: {extension}\")\n\n        data = parse_text_file_to_data(resolved_path, silent_errors)\n        self.status = data if data else \"No data\"\n        return data or Data()\n",
            "silent_errors_File-K3ohz": False,
            "chunk_overlap_CharacterTextSplitter-Zr2KP": 200,
            "chunk_size_CharacterTextSplitter-Zr2KP": 1000,
            "code_CharacterTextSplitter-Zr2KP": "from typing import List\n\nfrom langchain_text_splitters import CharacterTextSplitter\n\nfrom langflow.custom import CustomComponent\nfrom langflow.schema import Data\nfrom langflow.utils.util import unescape_string\n\n\nclass CharacterTextSplitterComponent(CustomComponent):\n    display_name = \"CharacterTextSplitter\"\n    description = \"Splitting text that looks at characters.\"\n\n    def build_config(self):\n        return {\n            \"inputs\": {\"display_name\": \"Input\", \"input_types\": [\"Document\", \"Data\"]},\n            \"chunk_overlap\": {\"display_name\": \"Chunk Overlap\", \"default\": 200},\n            \"chunk_size\": {\"display_name\": \"Chunk Size\", \"default\": 1000},\n            \"separator\": {\"display_name\": \"Separator\", \"default\": \"\\n\"},\n        }\n\n    def build(\n        self,\n        inputs: List[Data],\n        chunk_overlap: int = 200,\n        chunk_size: int = 1000,\n        separator: str = \"\\n\",\n    ) -> List[Data]:\n        # separator may come escaped from the frontend\n        separator = unescape_string(separator)\n        documents = []\n        for _input in inputs:\n            if isinstance(_input, Data):\n                documents.append(_input.to_lc_document())\n            else:\n                documents.append(_input)\n        docs = CharacterTextSplitter(\n            chunk_overlap=chunk_overlap,\n            chunk_size=chunk_size,\n            separator=separator,\n        ).split_documents(documents)\n        data = self.to_data(docs)\n        self.status = data\n        return data\n",
            "separator_CharacterTextSplitter-Zr2KP": " "
        },
        "GroupNode-3m6Jg": {
            "code_PromptComponent-KYRBR": "from langflow.base.prompts.api_utils import process_prompt_template\nfrom langflow.custom import Component\nfrom langflow.io import Output, PromptInput\nfrom langflow.schema.message import Message\nfrom langflow.template.utils import update_template_values\n\n\nclass PromptComponent(Component):\n    display_name: str = \"Prompt\"\n    description: str = \"Create a prompt template with dynamic variables.\"\n    icon = \"prompts\"\n    trace_type = \"prompt\"\n\n    inputs = [\n        PromptInput(name=\"template\", display_name=\"Template\"),\n    ]\n\n    outputs = [\n        Output(display_name=\"Prompt Message\", name=\"prompt\", method=\"build_prompt\"),\n    ]\n\n    async def build_prompt(\n        self,\n    ) -> Message:\n        prompt = await Message.from_template_and_variables(**self._attributes)\n        self.status = prompt.text\n        return prompt\n\n    def post_code_processing(self, new_build_config: dict, current_build_config: dict):\n        \"\"\"\n        This function is called after the code validation is done.\n        \"\"\"\n        frontend_node = super().post_code_processing(new_build_config, current_build_config)\n        template = frontend_node[\"template\"][\"template\"][\"value\"]\n        _ = process_prompt_template(\n            template=template,\n            name=\"template\",\n            custom_fields=frontend_node[\"custom_fields\"],\n            frontend_node_template=frontend_node[\"template\"],\n        )\n        # Now that template is updated, we need to grab any values that were set in the current_build_config\n        # and update the frontend_node with those values\n        update_template_values(frontend_template=frontend_node, raw_template=current_build_config[\"template\"])\n        return frontend_node\n",
            "template_PromptComponent-KYRBR": "Kindly provide a response to the user's inquiry, adhering to the provided context and message history. Please ensure the following rules are followed:\n\nAvoid repetition of information already stated in the context or message history.\nMaintain clarity and conciseness in your response.\nEnsure relevance to the user's question.\n\nContext: {context}\n\nMessage History:\n{history}\n\nUser's Question: {question}",
            "context_PromptComponent-KYRBR": "",
            "question_PromptComponent-KYRBR": "",
            "code_Memory-iGzuM": "from langflow.custom import Component\nfrom langflow.helpers.data import data_to_text\nfrom langflow.io import DropdownInput, IntInput, MessageTextInput, MultilineInput, Output\nfrom langflow.memory import get_messages\nfrom langflow.schema import Data\nfrom langflow.schema.message import Message\n\n\nclass MemoryComponent(Component):\n    display_name = \"Chat Memory\"\n    description = \"Retrieves stored chat messages.\"\n    icon = \"message-square-more\"\n\n    inputs = [\n        DropdownInput(\n            name=\"sender\",\n            display_name=\"Sender Type\",\n            options=[\"Machine\", \"User\", \"Machine and User\"],\n            value=\"Machine and User\",\n            info=\"Type of sender.\",\n            advanced=True,\n        ),\n        MessageTextInput(\n            name=\"sender_name\",\n            display_name=\"Sender Name\",\n            info=\"Name of the sender.\",\n            advanced=True,\n        ),\n        IntInput(\n            name=\"n_messages\",\n            display_name=\"Number of Messages\",\n            value=100,\n            info=\"Number of messages to retrieve.\",\n            advanced=True,\n        ),\n        MessageTextInput(\n            name=\"session_id\",\n            display_name=\"Session ID\",\n            info=\"Session ID of the chat history.\",\n            advanced=True,\n        ),\n        DropdownInput(\n            name=\"order\",\n            display_name=\"Order\",\n            options=[\"Ascending\", \"Descending\"],\n            value=\"Ascending\",\n            info=\"Order of the messages.\",\n            advanced=True,\n        ),\n        MultilineInput(\n            name=\"template\",\n            display_name=\"Template\",\n            info=\"The template to use for formatting the data. It can contain the keys {text}, {sender} or any other key in the message data.\",\n            value=\"{sender_name}: {text}\",\n            advanced=True,\n        ),\n    ]\n\n    outputs = [\n        Output(display_name=\"Chat History\", name=\"messages\", method=\"retrieve_messages\"),\n        Output(display_name=\"Messages (Text)\", name=\"messages_text\", method=\"retrieve_messages_as_text\"),\n    ]\n\n    def retrieve_messages(self) -> Data:\n        sender = self.sender\n        sender_name = self.sender_name\n        session_id = self.session_id\n        n_messages = self.n_messages\n        order = \"DESC\" if self.order == \"Descending\" else \"ASC\"\n\n        if sender == \"Machine and User\":\n            sender = None\n\n        messages = get_messages(\n            sender=sender,\n            sender_name=sender_name,\n            session_id=session_id,\n            limit=n_messages,\n            order=order,\n        )\n        self.status = messages\n        return messages\n\n    def retrieve_messages_as_text(self) -> Message:\n        messages_text = data_to_text(self.template, self.retrieve_messages())\n        self.status = messages_text\n        return Message(text=messages_text)\n",
            "n_messages_Memory-iGzuM": 100,
            "order_Memory-iGzuM": "Ascending",
            "sender_Memory-iGzuM": "Machine and User",
            "sender_name_Memory-iGzuM": "",
            "session_id_Memory-iGzuM": sessionID,
            "template_Memory-iGzuM": "{sender_name}: {text}"
        },
        "OllamaEmbeddings-qmIRW": {
            "base_url": "http://localhost:11434",
            "model": "llama3:latest",
            "temperature": 0
        },
        "TextInput-rNRx2": {
            "input_value": "This is the file explaintion\n" + feature_explaintion + "\n" + "Please read the explaintion and take look at \"x\",\"y\",\"imp\",\"co\". For every json object, x and y means two factors that x is a cause of y. The bigger imp(importance) value means x and y have strongger causation. If \"co\" value smaller than 0, that means the bigger x will make smaller y. If \"co\" (correlation) value bigger than 0, that means the bigger x will make bigger y.\n"
        },
        "APIRequest-yimsU": {
            "body": "{}",
            "curl": "",
            "headers": "{}",
            "method": "POST",
            "timeout": 5,
            "urls": ""
        },
        "ParseData-dEMZf": {
            "sep": "\n",
            "template": "{text}"
        }
    }
    
    if(model_type == "llama"):
        TWEAKS["APIRequest-yimsU"] = {
            "urls": "http://127.0.0.1:7860/api/v1/run/02fcf5f1-aeea-46a6-a534-596b3229b825?stream=false",
            "method": "POST",
            "headers": "{\"Content-Type\": \"application/json\"}",
            "body": "{\"input_value\": \"message\", \"output_type\": \"chat\", \"input_type\": \"chat\", \"tweaks\": {\"ChatInput-gDMtl\": {}, \"ChatOutput-T7brT\": {}, \"OpenAIModel-pASpz\": {}}}",
            "timeout": 5,
            "curl": ""
        }
    else:
        TWEAKS["APIRequest-yimsU"] = {
            "body": "{}",
            "curl": "",
            "headers": "{}",
            "method": "POST",
            "timeout": 5,
            "urls": ""
        }
    
    return TWEAKS



load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))

current_project_id = "-1"

sql_create()

# -----------------------------------------------UI--------------------------------------------------------#

st.title("Causal AI Chat Bot")

#feature_explaintion = st.text_area("Explain your csv file or features here (Not required)", placeholder="Type something")  # Can Store it to database

st.sidebar.header("Information to Access")

email = st.sidebar.text_input("Email", placeholder="Email")
password = st.sidebar.text_input("Password", placeholder="Password")
project_id = st.sidebar.text_input("Project ID", placeholder="Project ID")
sessionID = f"{email}-{project_id}"



if st.sidebar.button("Upload", use_container_width=True):
    request_json(email, password, project_id)
    if(current_project_id != project_id):
        load_session(sessionID)
        current_project_id = project_id


if os.path.exists(current_dir + "/json_file_storage" + f"/{email}/{project_id}.json"):
    save_dir = current_dir + "/json_file_storage" + f"/{email}/{project_id}.json"
    st.success("File uploaded and processed successfully!")
    
else:
    st.error("Please upload the json file first")
    

st.sidebar.header("Select for conversation")

# uploaded_file = st.sidebar.file_uploader("Choose a file for analyse...", 
#                                          type=['json'], help="Limit 1MB per file • json")

uploaded_explain = st.sidebar.file_uploader("Upload your metadata file [Optional]", 
                                         type=['txt'], help="Limit 1MB per file • txt")

model = st.sidebar.selectbox("Model", ["Llama3 8b", "GPT 4o", "GPT 4o mini"])
model_name, model_type = model_select(model)

sidebar = st.sidebar.container()
col1, col2 = sidebar.columns(2)

with col1:
    if st.button("Clear Dialogue", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("New Chat", type="primary", use_container_width=True):
        delete_session(sessionID)
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Ask me about the causal graph")

feature_explaintion = txt_to_string(uploaded_explain)
# ----------------------------------------Langflow----------------------------------------------------------#
BASE_API_URL = "http://127.0.0.1:7860"
FLOW_ID = "db2345d0-e4c8-4475-8528-bf1d994bc6e8"
ENDPOINT = "" 

TWEAKS = model_tweaks(model_type, model_name)


# -----------------------------------------------------------------------------------------------------------#

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        endpoint = ENDPOINT or FLOW_ID
        tweaks = TWEAKS
        api_key = None
        output_type = "chat"
        input_type = "chat"
    
        try:
            response = run_flow(
                message=prompt,
                endpoint=endpoint,
                output_type=output_type,
                input_type=input_type,
                tweaks=tweaks,
                api_key=api_key,
                session_id=sessionID,
            )

            if 'outputs' not in response:
                st.error(f"Unexpected API response structure. Response: {response}")
            else:
                try:
                    main_message = response["outputs"][0]["outputs"][0]["results"]["message"]["text"]
                    formatted_response = format_message(main_message)
                    st.markdown(formatted_response)
                    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                except KeyError as ke:
                    st.error(f"Error accessing response data: {ke}. Full response: {response}")
        except Exception as e:
            st.error(f"An error occurred while calling the API: {e}")

if __name__ == "__main__":
    pass