# config.py
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods


class Config:
    # IBM WatsonX
    IBM_CREDENTIALS = {
        "apikey": "NLpTAkvOrUXk61M1xV9h-PTwg4NbO359QKHCN11EJw0P",
        "url": "https://eu-de.ml.cloud.ibm.com",
        "project_id": "95e3936c-2b13-4aa4-a02a-2bf9da54e20b"
        
    }
    GOOGLE_CREDENTIALS = {

        "apikey": "AIzaSyAGtMcmPAeNQw-ndkPKCSEuQhWRJQZamyU"

    }
    
    IBM_PARAMS = {
        "temperature": 0.7,
        "max_new_tokens": 512,
        "top_p": 0.9,
        "top_k": 50
    }
    
    # PostgreSQL
    DB_PARAMS = {
        'host': 'localhost',
        'database': 'allam_ch',
        'user': 'postgres',
        'password': '1234',
        'port': '5433',
        
    }
    
    # Embedding Model
    EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
    EMBEDDING_DIMENSION = 768  # 

    PARAMETETRS = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 200,
        GenParams.STOP_SEQUENCES: ["<|endoftext|>"],
        GenParams.REPETITION_PENALTY: 1.0,
        GenParams.TEMPERATURE: 0.7 
    
    }