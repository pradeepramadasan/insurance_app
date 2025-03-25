from .iris import create_iris_agent
from .mnemosyne import create_mnemosyne_agent
from .ares import create_ares_agent
from .demeter import create_demeter_agent
from .apollo import create_apollo_agent
from .calliope import create_calliope_agent
from .plutus import create_plutus_agent
from .tyche import create_tyche_agent
from .orpheus import create_orpheus_agent
from .hestia import create_hestia_agent
from .dike import create_dike_agent
from .eirene import create_eirene_agent
from .themis import create_themis_agent
from .zeus import create_zeus_agent
from autogen import UserProxyAgent, GroupChat, GroupChatManager
from config import config_list_gpt4o

def initialize_agents():
    """Initialize all agents and return them as a dictionary"""
    iris = create_iris_agent()
    mnemosyne = create_mnemosyne_agent()
    ares = create_ares_agent()
    demeter = create_demeter_agent()
    apollo = create_apollo_agent()
    calliope = create_calliope_agent()
    plutus = create_plutus_agent()
    tyche = create_tyche_agent()
    orpheus = create_orpheus_agent()
    hestia = create_hestia_agent()
    dike = create_dike_agent()
    eirene = create_eirene_agent()
    themis = create_themis_agent()
    zeus = create_zeus_agent()
    
    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="ALWAYS",
        system_message="You are the human user interacting with the insurance multi-agent system."
    )
    
    return {
        "iris": iris,
        "mnemosyne": mnemosyne,
        "ares": ares,
        "demeter": demeter,
        "apollo": apollo,
        "calliope": calliope,
        "plutus": plutus,
        "tyche": tyche,
        "orpheus": orpheus,
        "hestia": hestia,
        "dike": dike,
        "eirene": eirene,
        "themis": themis,
        "zeus": zeus,
        "user_proxy": user_proxy
    }