import os
import requests
import logging
from typing import Dict, Any, Optional
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from langchain_openai import AzureChatOpenAI
from composio_langchain import ComposioToolSet
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TicketManagementAgent:
    def __init__(self):
        """Initialize the agent and validate environment variables."""
        load_dotenv()
        self._validate_env_vars()
        self.llm = self._initialize_llm()
        self.composio_toolset = ComposioToolSet(api_key=os.getenv("COMPOSIO_API_KEY"))
        self.agent_executor = self._setup_agent_executor()
        self.webhook_url = os.getenv("WEBHOOK_URL")
        self.task_classification_prompt = self._create_task_classification_prompt()

    def _validate_env_vars(self):
        """Validate required environment variables.""" 
        required_vars = [
            "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_MODEL", "AZURE_OPENAI_API_VERSION",
            "COMPOSIO_API_KEY", "WEBHOOK_URL", 
            "FRESHDESK_API_KEY", "FRESHDESK_DOMAIN"
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

    def _initialize_llm(self) -> AzureChatOpenAI:
        """Initialize Azure OpenAI Language Model."""
        return AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            model=os.getenv("AZURE_OPENAI_MODEL"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.7
        )

    def _create_task_classification_prompt(self) -> ChatPromptTemplate:
        """Create a prompt template for task classification.""" 
        return ChatPromptTemplate.from_template(""" 
        You are an intelligent task classifier. Analyze the user's input and determine the most appropriate action.

        User Input: {user_prompt}

        Possible Actions:
        1. create_ticket: When the user wants to create a new support ticket
        2. reply_ticket: When the user wants to respond to an existing ticket
        3. unknown: When the input doesn't clearly match the above actions

        Response Format:
        
json
        {{"action": "create_ticket" or "reply_ticket" or "unknown", "reasoning": "Brief explanation of why this action was chosen"}}
        """)

    def _setup_agent_executor(self) -> Optional[AgentExecutor]:
        """Setup the agent executor with available tools.""" 
        try:
            logging.info("Setting up agent executor...")
            prompt = hub.pull("hwchase17/openai-functions-agent")
            valid_actions = ['FRESHDESK_REPLY_TICKET', 'FRESHDESK_CREATE_TICKET']

            tools = self.composio_toolset.get_tools(actions=valid_actions)
            logging.info(f"Tools initialized: {tools}")
            agent = create_openai_functions_agent(self.llm, tools, prompt)
            logging.info("Agent executor successfully created.")
            return AgentExecutor(agent=agent, tools=tools, verbose=True)
        except Exception as e:
            logging.error(f"Error initializing agent executor: {e}")
            return None

    def _classify_task_with_llm(self, prompt: str) -> Dict[str, Any]:
        """Classify the task using LLM with a more robust approach.""" 
        try:
            classification_chain = self.task_classification_prompt | self.llm | JsonOutputParser()
            classification = classification_chain.invoke({"user_prompt": prompt})

            logging.info(f"Task classification result: {classification}")
            return {
                "action": classification.get("action", "unknown"),
                "reasoning": classification.get("reasoning", "No specific reasoning")
            }
        except Exception as e:
            logging.error(f"Error in task classification: {e}")
            return {"action": "unknown", "reasoning": str(e)}

    def _extract_ticket_details(self, prompt: str, action: str, existing_data: Dict[str, str]) -> Dict[str, Optional[str]]: 
        """Extract ticket details from the prompt based on the action and existing data.""" 
        default_return = {
            "ticket_id": existing_data.get("ticket_id"),
            "message": existing_data.get("message"),
            "description": existing_data.get("description"),
            "email": existing_data.get("email")
        }

        try:
            extraction_prompt = ChatPromptTemplate.from_template("""
            Extract ticket details from the following user prompt:

            Prompt: {prompt}
            Action: {action}

            Provide details in this JSON format:

json
            {{"ticket_id": "ticket ID if mentioned",
               "message": "reply message if mentioned",
               "description": "ticket description if mentioned",
               "email": "email address if mentioned"}}
            """)

            extraction_chain = extraction_prompt | self.llm | JsonOutputParser()
            extracted_details = extraction_chain.invoke({
                "prompt": prompt, 
                "action": action
            })

            for key in default_return:
                if extracted_details.get(key):
                    default_return[key] = extracted_details[key]

            return default_return

        except Exception as e:
            logging.warning(f"Detailed extraction failed: {e}. Falling back to existing data.")
            return default_return

    def _get_input_if_missing(self, field_name: str, prompt_message: str, existing_data: Dict[str, str]) -> Optional[str]:
        """Prompt the user for input if the required field is missing, skip if it's already provided.""" 
        if existing_data.get(field_name):
            return existing_data[field_name]

        field_value = input(prompt_message).strip()
        while not field_value:
            field_value = input(f"{field_name.capitalize()} cannot be empty. {prompt_message}").strip()
        return field_value

    def _execute_task(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]: 
        """Execute the task using the agent executor.""" 
        if not self.agent_executor:
            logging.error("Agent executor not initialized.")
            return None

        try:
            logging.info(f"Executing task: {task}")
            result = self.agent_executor.invoke({"input": task})
            logging.info(f"Task execution result: {result}")
            self._send_to_webhook(result)
            return result
        except Exception as e:
            logging.error(f"Error executing task: {e}")
            print("I'm sorry, something went wrong while processing your request. Could you try again?")
            return None

    def _send_to_webhook(self, data: Dict[str, Any]) -> None:
        """Send the task result to the webhook URL.""" 
        if not self.webhook_url:
            logging.error("No webhook URL configured.")
            return

        try:
            response = requests.post(self.webhook_url, json=data)
            if response.status_code == 200:
                logging.info("Successfully sent data to the webhook.")
            else:
                logging.warning(f"Webhook failed with status code: {response.status_code}")
        except Exception as e:
            logging.error(f"Error sending data to webhook: {e}")

    def process_task(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Process the user's prompt and execute the appropriate action.""" 
        existing_data = {}

        # Check for greeting or exit keywords
        if prompt.lower() in ["hi", "hello"]:
            print("Hello! How can I assist you with your ticket today?")
            return None
        elif prompt.lower() == "exit":
            print("Goodbye! Have a great day!")
            return None

        try:
            classification = self._classify_task_with_llm(prompt)
            action = classification['action']

            parsed_data = self._extract_ticket_details(prompt, action, existing_data)

            if action == 'reply_ticket':
                task_data = {
                    "action": "FRESHDESK_REPLY_TICKET",
                    "ticket_id": parsed_data.get("ticket_id"),
                    "message": parsed_data.get("message")
                }

                if not task_data["ticket_id"]:
                    task_data["ticket_id"] = self._get_input_if_missing(
                        "ticket_id", 
                        "Please provide the ticket ID to reply to:", 
                        task_data
                    )

                if not task_data["message"]:
                    task_data["message"] = self._get_input_if_missing(
                        "message", 
                        "What would you like to reply with?", 
                        task_data
                    )

                return self._execute_task(task_data)

            elif action == 'create_ticket':
                task_data = {
                    "action": "FRESHDESK_CREATE_TICKET",
                    "description": parsed_data.get("description"),
                    "email": parsed_data.get("email")
                }

                if not task_data["description"]:
                    task_data["description"] = self._get_input_if_missing(
                        "description", 
                        "Please provide a description for the new ticket:", 
                        task_data
                    )

                if not task_data["email"]:
                    task_data["email"] = self._get_input_if_missing(
                        "email", 
                        "Please provide your email address:", 
                        task_data
                    )

                return self._execute_task(task_data)

            else:
                print("I'm sorry, I am not able to answer that request. Please ask me to create or reply to a ticket.")
                return None
        except Exception as e:
            logging.error(f"Error processing task: {e}")
            return None

if __name__ == "__main__":
    agent = TicketManagementAgent()

    while True:
        user_prompt = input("How can I assist you with your ticket today? (Type 'exit' to end) ")
        agent.process_task(user_prompt)
        if user_prompt.lower() == "exit":
            break

