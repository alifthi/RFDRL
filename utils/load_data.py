import pandas as pd
from CONFIG import HH_LOAD, PRICE, PV_POWER, WT_POWER
class load_data:
    def __init__(self):
        self.load = self.load_csv(HH_LOAD)['power']
        self.price = self.load_csv(PRICE)['EUR/kWh']
        self.pv = self.load_csv(PV_POWER)['solar_power']
        self.wind = self.load_csv(WT_POWER)['wind_power']
        
    def load_csv(self, file_path):
        data = pd.read_csv(file_path)
        return self.preprocess_data(data)
    
    @staticmethod
    def preprocess_data(data):
        return data
    