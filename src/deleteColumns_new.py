
import pandas as pd

CSVPATH = "./data/all.csv"#

DROPLIST = ['Unnamed: 109', 'TAIL_NUM', 'FL_NUM', 'CARRIER', 'AIRLINE_ID', 'ORIGIN_STATE_ABR', 'DEST_STATE_ABR', 'DEST_CITY_NAME', 'ORIGIN_CITY_NAME', 'FL_DATE', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN_STATE_FIPS', 'ORIGIN_STATE_NM', 'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', 'DEST_STATE_FIPS', 'DEST_STATE_NM', 'DEST_WAC', 'DEP_DELAY_NEW', 'DEP_TIME_BLK', 'ARR_DELAY_NEW', 'ARR_TIME_BLK', 'CANCELLATION_CODE', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'FIRST_DEP_TIME', 'TOTAL_ADD_GTIME', 'LONGEST_ADD_GTIME', 'DIV_AIRPORT_LANDINGS', 'DIV_REACHED_DEST', 'DIV_ACTUAL_ELAPSED_TIME', 'DIV_ARR_DELAY', 'DIV_DISTANCE', 'DIV1_AIRPORT', 'DIV1_AIRPORT_ID', 'DIV1_AIRPORT_SEQ_ID', 'DIV1_WHEELS_ON', 'DIV1_TOTAL_GTIME', 'DIV1_LONGEST_GTIME', 'DIV1_WHEELS_OFF', 'DIV1_TAIL_NUM', 'DIV2_AIRPORT', 'DIV2_AIRPORT_ID', 'DIV2_AIRPORT_SEQ_ID', 'DIV2_WHEELS_ON', 'DIV2_TOTAL_GTIME', 'DIV2_LONGEST_GTIME', 'DIV2_WHEELS_OFF', 'DIV2_TAIL_NUM', 'DIV3_AIRPORT', 'DIV3_AIRPORT_ID', 'DIV3_AIRPORT_SEQ_ID', 'DIV3_WHEELS_ON', 'DIV3_TOTAL_GTIME', 'DIV3_LONGEST_GTIME', 'DIV3_WHEELS_OFF', 'DIV3_TAIL_NUM', 'DIV4_AIRPORT', 'DIV4_AIRPORT_ID', 'DIV4_AIRPORT_SEQ_ID', 'DIV4_WHEELS_ON', 'DIV4_TOTAL_GTIME', 'DIV4_LONGEST_GTIME', 'DIV4_WHEELS_OFF', 'DIV4_TAIL_NUM', 'DIV5_AIRPORT', 'DIV5_AIRPORT_ID', 'DIV5_AIRPORT_SEQ_ID', 'DIV5_WHEELS_ON', 'DIV5_TOTAL_GTIME', 'DIV5_LONGEST_GTIME', 'DIV5_WHEELS_OFF', 'DIV5_TAIL_NUM', 'CANCELLED', 'DIVERTED']

# 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER' not showing up

CHUNK_SIZE = 1000000
SAMPLE_RATE = 0.001
RANDOM_STATE = 1


def main(csvPath, droplist):
   #data = pandas.read_csv(csvPath, dtype={'user_id': int}, nrows=100, index_col=0)
   #headers = data.iloc[0]
   n = 1
   output = "output_sampled_1000.csv"
   skip = False
   
   random_state = RANDOM_STATE
    
   for chunk in pd.read_csv(csvPath, index_col=0,chunksize=CHUNK_SIZE):
       #print (n, chunk.iloc[0])
       #output = outFile + str(n) + ".csv"
       
       chunk = chunk[chunk['CANCELLED'] == False]
       chunk = chunk[chunk['DIVERTED'] == False]
       chunk = chunk.drop(DROPLIST, 1)
       
       chunk = chunk.dropna(axis=0)
       
       #count = len(chunk.index);
       #sample = count*SAMPLE_RATE
       
       
       chunk = chunk.sample(frac=SAMPLE_RATE, random_state = random_state)
       
       #sample = chunk.sample(frac = SAMPLE_RATE, random_state = random_state)

       #chunk = encode(chunk)
       if skip:
           chunk.to_csv(output, mode="a", header=False)
       else:
           chunk.to_csv(output, mode="a")
       skip = True
       
       random_state += 1
       n += 1
       print (n)
   
main(CSVPATH, DROPLIST)
