from Config import Config
import os
from langchain_community.document_loaders import TextLoader # Text loader
from langchain.text_splitter import CharacterTextSplitter # Text splitter
from langchain.prompts import ChatPromptTemplate # Chat prompt template
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser # Output parser
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from ibm_watsonx_ai.foundation_models import Model 
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain_ibm import WatsonxLLM
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, device='cpu')

    def embed_documents(self, texts):
        # Convert each numpy array to a list
        return [self.model.encode(text).tolist() for text in texts]

    def embed_query(self, text):
        # Convert the query embedding to a list
        return self.model.encode(text).tolist()
        

class rag_sys:
    def __init__(self):
        self.parameters = Config.PARAMETETRS
        self.vectorstore = self.create_vectordb('salwa.txt')
        self.watsonx_allam = WatsonxLLM(
            model_id='sdaia/allam-1-13b-instruct',
            url=Config.IBM_CREDENTIALS["url"],
            apikey=Config.IBM_CREDENTIALS["apikey"],
            project_id=Config.IBM_CREDENTIALS["project_id"],
            params=self.parameters
        )


    def get_access_token(self):
        token_url = "https://iam.cloud.ibm.com/identity/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": Config.IBM_CREDENTIALS['apikey']
        }
        
        try:
            response = requests.post(token_url, headers=headers, data=data, timeout=60)
            response.raise_for_status()
            return response.json()["access_token"]
        except RequestException as e:
            print(f"Error obtaining access token: {str(e)}")
            sys.exit(1)

    def rag_fun(self, user_query):
        prompt_template = """
            You are an Arabic language assistant. You must ALWAYS respond in Arabic regardless of the input language.
            Always maintain formal Arabic (فصحى) in your responses.
            
            Context: {context}
            
            Question: {question}
            
            Important Instructions:
            - Respond ONLY in Arabic
            - Give a precise and brief answer
            - Give the required answer and not extends this to more details 
            - If you're unsure, say "لا أعلم" (I don't know)
            - Maintain Arabic writing style and grammar
            - Pharaphrase the answer in interactive style as human-like style
            
            Arabic Response:"""
            
        PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        # retriving process
        qa = RetrievalQA.from_chain_type(llm=self.watsonx_allam, 
                                 chain_type="stuff", 
                                 retriever=self.vectorstore.as_retriever(),
                                 chain_type_kwargs={
                                    "prompt": PROMPT,
                                    "verbose": False
                                },
                                return_source_documents=False
                                )

        res = qa.invoke(user_query)
        return res['result']


    def create_vectordb(self, file_name):
        # Load the document
        loader = TextLoader(file_name)
        documents = loader.load()
        
        # Initialize the text splitter
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        # Split the document into chunks
        chunks = text_splitter.split_documents(documents)

        # embeddings models 
        embedding_model = SentenceTransformerEmbeddings("intfloat/multilingual-e5-large-instruct")

        # Create the vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory='./chroma_db'
        )

        return vectorstore

        


    def paraphrase_human_like(self, text):

        #text = self.rag_fun(text)

        token_access = self.get_access_token()

        url = "https://eu-de.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
        body = {
            "input": f"""paraphrase the following text in arabic without adding new information and stick to the original text length and you have to change words and order of sentences if possible , please just give one input and don't give new example 
        
        
        Input: تتميز العمارة التقليدية في المنطقة العربية باستخدام مواد البناء المحلية والتكيف مع الظروف المناخية القاسية. فقد طور المعماريون القدماء تقنيات مبتكرة مثل الملاقف الهوائية والأفنية الداخلية للتحكم في درجات الحرارة. كما استخدموا الزخارف الهندسية والنباتية لتجميل المباني وإضفاء الطابع الجمالي عليها.
        Output: يمكن للسياح زيارة قصر سلوى التاريخي المتواجد في منطقة الطريف بالدرعية من خلال الرحلات المنظمة. غير أن إمكانية الدخول إلى كافة الأقسام والغرف الداخلية تخضع لتنظيمات الجهة المشرفة على المكان. وللتأكد من المناطق التي يُسمح بزيارتها داخل القصر، يُفضل الاتصال بهيئة تطوير بوابة الدرعية أو مراجعة موقعهم على الإنترنت للحصول على التفاصيل الكاملة
        
        Input: يعتمد الطب التقليدي على استخدام الأعشاب الطبيعية والممارسات العلاجية المتوارثة عبر الأجيال. وقد وثق العلماء العرب القدماء خصائص مئات النباتات الطبية وطرق استخدامها في علاج الأمراض المختلفة. ما زالت هذه المعرفة تشكل أساساً مهماً للعديد من الأدوية الحديثة
        Output: أتعرف سر الشفاء الذي تناقلته الأجيال؟ إنه كنز من المعرفة الطبية الأصيلة التي جمعها أجدادنا عن النباتات الشافية من حولنا! كل عشبة تحمل قصة، وكل وصفة تحمل خبرة مئات السنين. والمدهش أن كثيراً من أدويتنا اليوم مستوحاة من هذه المعرفة القديمة التي دونها علماؤنا بعناية
        
        Input: تشتهر المنطقة العربية بتنوع حرفها اليدوية التقليدية، من النسيج إلى صناعة الفخار والنحاس. تتوارث العائلات هذه المهارات عبر الأجيال، محافظة على أساليب الإنتاج التقليدية. تواجه هذه الحرف اليوم تحديات المنافسة مع المنتجات الصناعية الحديثة
        
        Output: دعني أخذك في رحلة إلى عالم الحرف اليدوية الساحر! تخيل أنامل ماهرة تحوك الخيوط، وأيادٍ خبيرة تشكل الطين والنحاس بإتقان. هذه ليست مجرد مهن - إنها إرث عائلي ثمين يتناقله الآباء والأمهات لأبنائهم بفخر. صحيح أن المصانع الحديثة تنافس هذه الحرف، لكن يظل للمنتج اليدوي سحره الخاص الذي لا يضاهى
        
        Input: {text}
        
        Output: 
            """, 
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": 100,
                    "repetition_penalty": 1.05
                },
            "model_id": "sdaia/allam-1-13b-instruct",
            "project_id": Config.IBM_CREDENTIALS["project_id"]
        }
        
    
        headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer " +token_access
        }
    
        response = requests.post(
            url, 
            headers=headers, 
            json=body
        )
    
        if response. status_code!= 200:
            raise Exception("Non-200 response: " + str(response.text))
        data = response.json()
    
        
        return data['results'][0]['generated_text']
        