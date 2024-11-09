from google_guild import google_guide
from rag_sys import rag_sys


class Main_class:
    def __init__(self):


    def main(user_request, lat, lag, a_lat_1, a_lag_1, b_lat_2, b_lat_2):
        google_g = google_guide(user_request, lat, lag)
        
        # classify the query 
        class_v = google_g.allam_model_classify()

        if class_v == 1:
            ## welocme you are in turaif 
            
            # code of the google api for current location /

            # direction and position
            
            # main landmark 
            
            pass

        elif class_v == 2:

            # specific place type

            # return the closer one 

            # give the assistant template to talk about the 
            pass

        elif class_v == 3:

            #specific location name 
            pass 

        elif class_v == 4:

            # 

            pass

        else:

            sys_rag = rag_sys()
            #text = sys_rag.rag_fun(user_request)
            text = sys_rag.paraphrase_human_like(user_request)

            # here voice role 


        
        


if __name__ == "__main__":

    user_request = "  "

    user_lat = 24.734
    user_lag = 46.573

    area_lat_1, area_lag_1 = 24.730, 46.570  # bottom-left corner of area
    area_lat_2, area_lag_2 = 24.740, 46.580  # top-right corner of area
    
    
    MyClass.main(user_request, user_lat, user_lag, area_lat_1, area_lag_1, area_lat_2, area_lag_2)