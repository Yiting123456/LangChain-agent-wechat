
# 导入
from wxauto import WeChat
import time
import os
from dotenv import load_dotenv
from langchain import hub

from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import tools
import uiautomation
import win32gui
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from tools.tools import get_tag_values_tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='langchain_core.messages')

chat_memories = {}

def ollama_reply(chat_id, message):
    """为指定的用户获取或创建 Agent 并生成回复"""
    if chat_id not in chat_memories:
        chat_memories[chat_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    memory = chat_memories[chat_id]

    base_prompt  = hub.pull("hwchase17/react-chat") 

    llm = ChatOllama(base_url="http://localhost:11434", model="qwen3:14b", temperature=0.1)
    tools = [get_tag_values_tool] 

    agent = create_react_agent(llm, tools, base_prompt)

    agent_executor = AgentExecutor(agent=agent, 
                                   tools=tools,
                                   memory=memory,
                                   verbose=True,
                                   max_iterations=5,
                                   return_intermediate_steps=False,
                                   handle_parsing_errors=True)

    response = agent_executor.invoke({"input": message})
    AIreplay = response.get('output')
    return AIreplay


def my_callback(msg, chat):
    """
    msg: 当前收到的消息对象
    chat: Chat 实例，代表该消息所属的聊天窗口
    """
    if msg.sender == 'Promises.':
        print("*"*100)
        print(msg.sender, msg.content)
        reply = ollama_reply(chat.who, msg.content)
        print("{}".format(reply))
        print("*"*100)
        chat.SendMsg(reply)


wx = WeChat(debug=True)

wx.AddListenChat("Promises.", my_callback)

wx.KeepRunning()
