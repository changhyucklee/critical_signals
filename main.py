import pandas as pd
import PIconnect as PI
import datetime
import pyodbc
# import requests
import os
import time
from misc import parameters as p
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
def get_df_mes(sql):
    conn = pyodbc.connect(p.conn_string)
    df_input = pd.read_sql(sql, conn)
    conn.close()
    return df_input


def PItag_to_Datframe(tag_list, start, end, freq):
    PI.PIConfig.DEFAULT_TIMEZONE = 'Asia/Seoul'
    with PI.PIServer() as server:
        data_all = pd.DataFrame()

        for tag in tag_list:
            points = server.search(tag)[0]
            data = points.interpolated_values(start_time=start, end_time=end, interval=freq)
            data = pd.to_numeric(data, errors='coerce')
            data_all = pd.concat([data_all, data], axis=1)

    return data_all

def AFdata_to_Dataframe(MMU,Machine,af_list, start, end, freq):
    PI.PIConfig.DEFAULT_TIMEZONE='Asia/Seoul'
    import pandas as pd
    data_all = pd.DataFrame()
    database = PI.PIAFDatabase(server='YEJRC1AF01',database="YJ_Global_Remelt_Recycling")
    for af in af_list:
        Cat = af.split('|')[0]
        Asset = af.split('|')[1]
        attribute = database.children[MMU].children[Machine].attributes[Cat].children[Asset]
        data = attribute.interpolated_values(start_time=start, end_time=end, interval= freq)
        data_all = pd.concat([data_all, data], axis=1)
    data_all.insert(0, 'Machine', Machine)
    return data_all

def INIT_TABLE():
    try:
        # Create the connection
        conn = pyodbc.connect(p.Azure_conn_string)
        cursor = conn.cursor()
        sql = """
            DELETE FROM [dbo].[yej_decoater_ubc];
            DELETE FROM [dbo].[yej_decoater_lng];
        """
        cursor.execute(sql)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as err:
        logging.info({"Error": str(err)})
def INS_UBC(data_df):

    try:
        conn = pyodbc.connect(p.Azure_conn_string)
        cursor = conn.cursor()
        sql = """
        IF NOT EXISTS (SELECT * FROM [dbo].[yej_decoater_ubc] WHERE [SUMDAY] = ? AND [DROPNO] = ? )
            INSERT INTO [dbo].[yej_decoater_ubc]([SUMDAY],[DROPNO],[ALLOY],[UBC_WGT],[C1_WGT],[C2_WGT],[C3_WGT]
            ,[HOLD_DROSS],[DRS_WHITE],[DRS_BLACK]
            ,[SALT],[LOSS_A],[LOSS_B],[LOSS_C],[LOSS_D],[LOSS_E],[LOSS_F]) values (?,?,?,?,? ,?,?,?,?,? ,?,?,?,?,? ,?,?)
        """
        for index, row in data_df.iterrows():
            cursor.execute(sql, row['SUMDAY'],row['DROPNO'], row['SUMDAY'], row['DROPNO'], row['ALLOY']
                           , row['UBC_WGT'] ,row['C1_WGT'],row['C2_WGT'], row['C3_WGT'], row['HOLD_DROSS'],row['DRS_WHITE'],row['DRS_BLACK']
                           , row['SALT'],row['LOSS_A'], row['LOSS_B'], row['LOSS_C'], row['LOSS_D'], row['LOSS_E'], row['LOSS_F'])
            conn.commit()
        cursor.close()
        conn.close()
    except Exception as err:
        logging.info({"Error": str(err)})

def INS_LNG(data_df):

    try:
        conn = pyodbc.connect(p.Azure_conn_string)
        cursor = conn.cursor()
        sql = """
        IF NOT EXISTS (SELECT * FROM [dbo].[yej_decoater_lng] WHERE [SUMDAY] = ?)
            INSERT INTO [dbo].[yej_decoater_lng]([SUMDAY],[RK1_A],[RK1_B],[RK2_A],[RK2_B],[SM1_A],[SM1_B],[SM2_A],[SM2_B]
            ,[SM3_A] ,[SM3_B],[TM_A],[TM_B],[TH_A],[TH_B],[DBF_A] ,[DBF_B]
            ) values (?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?)
        """

        for index, row in data_df.iterrows():
            cursor.execute(sql, row['SUMDAY'],row['SUMDAY'],row['RK1_A'], row['RK1_B'], row['RK2_A'], row['RK2_B'],row['SM1_A'], row['SM1_B'], row['SM2_A'], row['SM2_B']
                           , row['SM3_A'], row['SM3_B'], row['TM_A'], row['TM_B'],row['TH_A'],row['TH_B'],row['DBF_A'],row['DBF_B'])
            conn.commit()
        cursor.close()
        conn.close()
    except Exception as err:
        logging.info({"Error": str(err)})

def INS_Water(data_df):

    try:
        conn = pyodbc.connect(p.Azure_conn_string)
        cursor = conn.cursor()
        sql = """
        IF NOT EXISTS (SELECT * FROM [dbo].[yej_decoater_Water_Air] WHERE [_Timestamp] = ?)
            INSERT INTO [dbo].[yej_decoater_Water_Air]([_Timestamp], [Machine]
            ,[WaterFlow_Afterburner_dg],[WaterFlow_Duct_dg],[WaterFlow_Kiln_dg],[Discharge_Airlock_Fault_dg]
            ) values (?,?,?,?,?, ?)
        """

        for index, row in data_df.iterrows():
            cursor.execute(sql, row['_Timestamp'],row['_Timestamp'],row['Machine']
                           ,row['WaterFlow_Afterburner_dg'], row['WaterFlow_Duct_dg'], row['WaterFlow_Kiln_dg'], row['Discharge_Airlock_Fault_dg']
               )
            conn.commit()
        cursor.close()
        conn.close()
    except Exception as err:
        logging.info({"Error": str(err)})

def INS_RunTime(data_df):
    try:
        conn = pyodbc.connect(p.Azure_conn_string)
        cursor = conn.cursor()
        sql = """
        IF NOT EXISTS (SELECT * FROM [dbo].[yej_decoater_RunTime] WHERE [To_Timestamp] = ? and [Machine] = ?)
            INSERT INTO [dbo].[yej_decoater_RunTime]([Day],[From_Timestamp],[To_Timestamp], [Machine]
            ,[Decoater_Day_Running_Time_Utilization_2_dg],[Decoater_Day_Productivity_Avg_2_dg]
            ) values (?,?,?,?,?,?)
        """

        for index, row in data_df.iterrows():
            cursor.execute(sql, row['To_Timestamp'],row['Machine'],row['Day'],row['From_Timestamp'],row['To_Timestamp'],row['Machine']
                           ,row['Decoater_Day_Running_Time_Utilization_2_dg'], row['Decoater_Day_Productivity_Avg_2_dg'])
            conn.commit()
        cursor.close()
        conn.close()
    except Exception as err:
        logging.info({"Error": str(err)})

def main():
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    logging.info('###################### Start !! ################')
    # logging.info('###################### INIT_TABLE ################')
    # INIT_TABLE()

    # df_ubc = get_df_mes(p.sql_ubc)
    # logging.info('INS_UBC')
    # INS_UBC(df_ubc)

    # df_lng = get_df_mes(p.sql_lng)
    # logging.info('INS_LNG')
    # INS_LNG(df_lng)
    today = str(today) + ' 06:30:00'
    yesterday = str(yesterday) + ' 06:30:00'
    # today = '2021-06-23 06:30:00'
    # yesterday = '2021-06-22 06:30:00'
    #
    # # df_Water = PItag_to_Datframe(p.tag_list_10s, yesterday, today, '10s')
    # df_Water = pd.DataFrame()
    # for Machine in p.Machine_list:
    #     df_Water_t = AFdata_to_Dataframe('Recycling', Machine, p.af_list_10s, yesterday, today, '10s')
    #     df_Water = pd.concat([df_Water, df_Water_t], axis=0)
    #
    # df_Water.columns = [x.replace(' ','_') for x in df_Water.columns]
    # df_Water = df_Water.reset_index()
    # df_Water.rename(columns={'index': '_Timestamp'}, inplace=True)
    # df_Water['Discharge_Airlock_Fault_dg'] = df_Water['Discharge_Airlock_Fault_dg'].astype(str)
    # df_Water['Discharge_Airlock_Fault_dg'].replace({'False': 0, 'True': 1}, inplace=True) #0:미발생 / 1:발생
    # logging.info('INS_Water')
    # INS_Water(df_Water.head(60 * 6 * 12))
    # INS_Water(df_Water.tail(60*6*12))
    #
    # df_RunTime = PItag_to_Datframe(p.tag_list_24h, yesterday, today, '24h')
    # day_list= ['2021-04-01',
    #  '2021-04-02',
    #  '2021-04-03',
    #  '2021-04-04',
    #  '2021-04-05',
    #  '2021-04-06',
    #  '2021-04-07',
    #  '2021-04-08',
    #  '2021-04-09',
    #  '2021-04-10',
    #  '2021-04-11',
    #  '2021-04-12',
    #  '2021-04-13',
    #  '2021-04-14',
    #  '2021-04-15',
    #  '2021-04-16',
    #  '2021-04-17',
    #  '2021-04-18',
    #  '2021-04-19',
    #  '2021-04-20',
    #  '2021-04-21',
    #  '2021-04-22',
    #  '2021-04-23',
    #  '2021-04-24',
    #  '2021-04-25',
    #  '2021-04-26',
    #  '2021-04-27']
    #
    # day_list = ['2022-04-30',
    #  '2022-05-01',
    # ]
    # for i in range(len(day_list)):
    #     yesterday=day_list[i]+' 06:30:00'
    #     if day_list[i] == '2022-05-01':
    #         break
    #     else:
    #         today=day_list[i+1]+' 06:30:00'
    df_RunTime = pd.DataFrame()
    for Machine in p.Machine_list:
        df_RunTime_t = AFdata_to_Dataframe('Recycling', Machine, p.af_list_24h, yesterday, today, '24h')
        # df_RunTime_t['Decoater_Day_Running_Time_Utilization_2_dg'] = df_RunTime_t['Decoater_Day_Running_Time_Utilization_2_dg'].round(1)
        # df_RunTime_t['Decoater_Day_Productivity_Avg_2_dg'] = df_RunTime_t['Decoater_Day_Productivity_Avg_2_dg'].round(2)
        df_RunTime = pd.concat([df_RunTime, df_RunTime_t.tail(1)], axis=0)
    df_RunTime.columns = [x.replace(' ', '_') for x in df_RunTime.columns]
    df_RunTime['From_Timestamp'] = yesterday
    df_RunTime['To_Timestamp'] = today
    df_RunTime['Day'] = df_RunTime['From_Timestamp'].str.slice(0,11)
    df_RunTime = df_RunTime.reset_index()
    df_RunTime.rename(columns={'index': '_Timestamp'}, inplace=True)
    logging.info('INS_RunTime')
    INS_RunTime(df_RunTime)

if __name__ == "__main__":
    main()
