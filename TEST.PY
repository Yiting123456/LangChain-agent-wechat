
# 导入
from wxauto import WeChat
import time
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.tools import Tool  
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import uiautomation
import win32gui
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from tools.tools_test import get_tag_values_tool,get_sim_ID_tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='langchain_core.messages')

chat_memories = {}

llm = ChatOllama(base_url="http://localhost:11434", model="gemma3:4b", temperature=0.5)


tools = [get_sim_ID_tool,get_tag_values_tool] 

custom_template = """Assistant is a large language model trained for Andritz Metris Platform and default to replay in Chinese.

Assistant is designed as a helpful factory data assistant. It can obtain the most likely tag IDs through user descriptions, then retrieve corresponding tag values via IDs, and also assist with a wide range of tasks—from answering simple questions to providing in-depth explanations and discussions on various topics. As a language model, Assistant can generate human-like text based on input, enabling natural conversations and coherent, topic-relevant responses.

Assistant is constantly learning and improving, with evolving capabilities. It can process and understand large amounts of text, using this knowledge to provide accurate and informative answers. Additionally, it can generate original text based on input, supporting discussions, explanations, and descriptions across diverse topics.

Overall, Assistant is a powerful tool for factory data-related tasks and beyond, offering valuable insights on various topics. Whether you need help with a specific question or want to discuss a topic, Assistant is here to assist.

TOOLS:
------

Assistant has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}

"""
base_prompt = ChatPromptTemplate.from_template(
    template=custom_template,
)

agent = create_react_agent(llm, tools, base_prompt)

def ollama_reply(chat_id, message):
    """为指定的用户获取或创建 Agent 并生成回复"""
    if chat_id not in chat_memories:
        chat_memories[chat_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    memory = chat_memories[chat_id]

    agent_executor = AgentExecutor(agent=agent, 
                                   tools=tools,
                                   memory=memory,
                                   verbose=True,
                                   max_iterations=8,
                                   return_intermediate_steps=False,
                                   handle_parsing_errors=True)

    response = agent_executor.invoke({"input": message})

    AIreplay = response.get('output')
    print('memory:',memory)
    print("AI Reply:", AIreplay)
    return AIreplay

people = 'Cheng'
def my_callback(msg, chat):
    """
    msg: 当前收到的消息对象
    chat: Chat 实例，代表该消息所属的聊天窗口
    """
    if msg.sender == people:
        print("*"*100)
        print(msg.sender, msg.content)
        reply = ollama_reply(chat.who, msg.content)
        print("{}".format(reply))
        print("*"*100)
        chat.SendMsg(reply)


wx = WeChat(debug=True)

wx.AddListenChat(people, my_callback)
#wx.AddListenChat("Cheng", my_callback)

wx.KeepRunning()