import requests
from Config import Config

class google_guide:
    def __init__(self, user_lat, user_lag, user_query):
        self.user_lat = user_lat 
        self.user_lag = user_lag 

    def allam_model_classify(self):
        # url 
        url = "https://eu-de.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
        
        # body
        body = {
            
     "input": f"""
    
            صنف استعلام المستخدم الى احد الاصناف التالية وذلك باختيار رقم التصنيف المقصود:
            واسترجاع العنصر المكاني كما في الامثلة
            ١. الاستعلام عن مكان المستخدم
            ٢. الاستعلام عن أقرب مكان بذكر نوع المكان دون اسمه كالمطعم والمدرسة
            ٣. الاستعلام عن أفضل مكان وفقا لآراء الآخرين بذكر نوع المكان دون اسمه  
            ٤. الاستعلام عن مكان باسمه كذكر اسم المطعم او المدرسة
            ٥. الاستعلام عن معلومات عامة عن أي مكان ( وصفا أو تاريخا أو غير ذلك من التفاصيل )
             فضلا اختيار تصنيف واحد وهو الاكثر احتمالية دون الادلاء بأي تفاصيل تتعلق بالمكان او الاجابة على بعض طلب المستخدم 
    
    
            Input: أين أنا الآن
            Output: ١ - لايوجد
    
            Input: ما هي المنطقة التي أنا فيها
            Output: ١ - لايوجد
    
            Input: ما هو موقعي الحالي
            Output: ١ - لايوجد
    
            Input: كم يبعد عني قصر سلوى
            Output: ٣ - قصر سلوى
    
            Input: ما أهم المعالم المحيطة بي
            Output: ٢ - المعالم
    
            Input: ما أقرب كافي لموقعي الحالي
            Output: ٢ - كافي
    
            Input: ما أقرب منطقة أثرية
            Output: ٢ - منطقة أثرية
    
            Input: ما تاريخ الدرعية
            Output: ٥ - لايوجد
    
            Input: ما قصة هذا المسجد
            Output: ٥ - لايوجد
    
            Input: لماذا لم يتم ترميم هذا المبنى
            Output: ٥ - لايوجد
    
            Input: أين يقع مسجد طريف
            Output: ٤ - مسجد طريف
            
            Input: اين تقع مدرسة الامجاد
            Output: ٤ - مدرسة الامجاد
            
            Input: ما أقرب حديقة
            Output: ٢ - حديقة
            
            Input: ما هي افضل مدرسة هنا
            Output : ٣ - مدرسة
    
            Input: ما هو افضل مطعم قريب من هنا
            Output : ٣ - مطعم
    
            Input: {query}
            Output:""", 
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": 10,
                    "repetition_penalty": 2,
                    "temprature" : 0.3, 
                    "top_k" : 2, 
                    "top_p" : 0.9
                },
            "model_id": "sdaia/allam-1-13b-instruct",
            "project_id": Config.IBM_CREDENTIALS['project_id']
            
        }
        
        # headers 
        headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer " +token_access
        }
        
        # response 
        response = requests.post(
            url, 
            headers=headers, 
            json=body
        )
    
        if response. status_code!= 200:
            raise Exception("Non-200 response: " + str(response.text))
        data = response.json()
    
           
        return data.get('results', [{}])[0].get('generated_text', "No text generated")

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
            
    def check_current_location(point, bottom_left, top_right):
        """
        Determine if the point is within bounds and identify its location.
        Returns:
            - "Inside: Center/North/South/East/West" if within bounds
            - "Outside: North/East/South/West" if outside bounds
        """
        lat, lon = point
        lat_min, lon_min = bottom_left
        lat_max, lon_max = top_right
        
        # Check if the point is inside bounds
        inside_lat = lat_min <= lat <= lat_max
        inside_lon = lon_min <= lon <= lon_max
        
        if inside_lat and inside_lon:
            # Calculate midpoints for center classification
            mid_lat = (lat_min + lat_max) / 2
            mid_lon = (lon_min + lon_max) / 2
            
            # Identify the internal position
            if abs(lat - mid_lat) < 0.001 and abs(lon - mid_lon) < 0.001:
                return "Inside: Center"
            elif lat > mid_lat:
                return "Inside: North"
            elif lat < mid_lat:
                return "Inside: South"
            elif lon > mid_lon:
                return "Inside: East"
            else:
                return "Inside: West"
        else:
            # Outside bounds - determine direction
            if lat > lat_max:
                return "Outside: North"
            elif lat < lat_min:
                return "Outside: South"
            elif lon > lon_max:
                return "Outside: East"
            elif lon < lon_min:
                return "Outside: West"
    
