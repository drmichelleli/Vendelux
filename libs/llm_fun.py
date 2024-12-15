import dotenv
import os
import logging
import json
import openai
import asyncio
import ssl
from aiohttp import ClientSession, TCPConnector

dotenv.load_dotenv()
from functools import partial
from tenacity import retry, stop_after_delay, wait_fixed, retry_if_exception_type
import logging
from datetime import datetime
retry_on_rate_limit_error = partial(
    retry,
    stop=stop_after_delay(300),  
    wait=wait_fixed(10),   
    retry=retry_if_exception_type(openai.error.RateLimitError),
)()

# define logger,file handler and set formatter
logger = logging.getLogger(__name__)
log_file = datetime.now().strftime('%Y_%m_%d_%H_%M')+'_llm_agent.log'
filename='./Logs/llm/'+ log_file
file_handler = logging.FileHandler(filename)
logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[file_handler],
)

class LLMScore():

    def __init__(self,text,prompt_template):
        self.gpt_model_type_4k = os.environ["AZURE_OPENAI_GPT35_MODEL_NAME1"]
        self.gpt_model_type_16k = os.environ["AZURE_OPENAI_GPT35_MODEL_NAME"]
        openai.api_key = os.environ["OPENAI_API_KEY"]
        openai.api_base = os.environ["OPENAI_ENDPOINT"]
        openai.api_version = os.environ["OPENAI_API_VERSION"]
        openai.api_type = os.environ["OPENAI_API_TYPE"]
        self.text=text
        self.prompt_template=prompt_template
        self.responses=[]
        self.counter_4k=0
        self.counter_16k=0
        

    def function(self,row):
        prompt=self.prompt_template(row)
        return prompt

    
    @retry_on_rate_limit_error
    async def get_completion_async(self,row,gpt_model_4k,gpt_model_16k,record_id):
        prompt=self.function(row)
        
        messages = [{"role": "system", "content": "You are a helpful text analysis tool."},
            {"role": "user", "content": prompt}]
        
        try:
            ssl_ctx = ssl.create_default_context(cafile=os.environ.get("REQUESTS_CA_BUNDLE", None))
            conn = TCPConnector(ssl_context=ssl_ctx)
            session = ClientSession(connector=conn,trust_env=True)
            openai.aiosession.set(session)

            response =await openai.ChatCompletion.acreate(
                engine=gpt_model_4k,
                messages=messages,
                temperature=0, # this is the degree of randomness of the model's output
            )
            tp=json.loads(response.choices[0].message['content'])
            tp['id']=record_id
            self.responses.append(tp)
            self.counter_4k+=1

            await openai.aiosession.get().close()

        except openai.error.RateLimitError as e:

            print("Rate limit exceeded. Retrying in 30 seconds...")
            await openai.aiosession.get().close()

            raise
        except openai.error.InvalidRequestError as e:
            try:
                print("Moving to the 16k model")
                response =await openai.ChatCompletion.acreate(
                    engine=gpt_model_16k,
                    messages=messages,

                    temperature=0, # this is the degree of randomness of the model's output
                )

                tp=json.loads(response.choices[0].message['content'])
                tp['id']=record_id
                self.responses.append(tp)
                self.counter_16k+=1

                await openai.aiosession.get().close()
            except Exception as e:
                logging.exception('')
                await openai.aiosession.get().close()
                pass
            
        except Exception as e:
            logging.exception('')
            await openai.aiosession.get().close()
  
            pass

    def build_coros(self):
            coros = [self.get_completion_async(
            self.text.iloc[i]["text"],
            self.gpt_model_type_4k,
            self.gpt_model_type_16k,
            self.text.iloc[i]["id"],
        )
        for i in range(len(self.text))
    ]
            return coros
            
    async def gather_with_concurrency(self,n, *coros):
        self.responses=[]
        semaphore = asyncio.Semaphore(n)
        logging.info("Starting scoring")
        logging.info("To be scored %s calls", len(self.text))

        async def sem_coro(coro):
            async with semaphore:
                return await coro
        return await asyncio.gather(*(sem_coro(c) for c in coros))
    

    
    async def run(self):
        coros=self.build_coros()
        await  self.gather_with_concurrency(100, *coros)



    def return_results(self):
        logging.info("Sent to the 4k model: %s",self.counter_4k)
        logging.info("Sent to the 16k model: %s",self.counter_16k)
        return self.responses