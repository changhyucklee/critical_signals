import Utils as U
import datetime
from misc import parameters as p

import pandas as pd
import numpy as np
import pyodbc

import os
import pathlib
import logging
import warnings
import time

import schedule

# log 출력 형식
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG,  format=formatter)
# # 로그 생성
logger = logging.getLogger()
#
warnings.filterwarnings("ignore")
# # log를 파일에 출력
# file_handler = logging.FileHandler('my.log')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)


today = datetime.date.today()
StartDay = today - datetime.timedelta(days=8)
EndDay = today - datetime.timedelta(days=1)
StartDay_3_1 = today - datetime.timedelta(days=3)
StartDay_1 = today - datetime.timedelta(days=1)
StartDay_2 = today - datetime.timedelta(days=2)

MonthStartDay = today - datetime.timedelta(days=31)
MonthStartDay = MonthStartDay.strftime('%Y-%m-%d')+' 06:30:00'

OneYearAgoDay = today - datetime.timedelta(days=365)
OneYearAgoDay = OneYearAgoDay.strftime('%Y-%m-%d')+' 06:30:00'

ToDay_0630 = today.strftime('%Y-%m-%d')+' 06:30:00'

StartDay2 = StartDay.strftime('%Y-%m-%d')+' 06:30:00'
EndDay2 = EndDay.strftime('%Y-%m-%d')+' 06:30:00'
StartDay_3 = StartDay_3_1.strftime('%Y-%m-%d') + ' 06:30:00'
StartDay_1_1 = StartDay_1.strftime('%Y-%m-%d') + ' 06:30:00'
StartDay_2_1 = StartDay_2.strftime('%Y-%m-%d') + ' 06:30:00'
# Get lists of dates where Out was detected
def get_date_Out(df,out_cols):
    check_cols = ['check_' + col for col in out_cols]
    cnt_Out = df.groupby('just_date').agg({'BATCHNO': 'count', **{col: 'sum' for col in check_cols}})
    for col in out_cols:
        cnt_Out['Out_' + col] = cnt_Out.apply(lambda x: 'Out' if x['check_' + col] >= x['BATCHNO'] / 2.5 else 'In',
                                              axis=1)

    # Get lists of dates where Out was detected for each column
    date_Out = {col: cnt_Out.loc[cnt_Out['Out_' + col] == 'Out'].index.to_list() for col in out_cols}
    return date_Out

# get graph and statement
def get_data(lst_Target,date_Out, avg_values,aDF):
    if aDF is None:
        return  [], None
    else:
        plot_type_map = U.get_plot_type_map(lst_Target, date_Out, avg_values)
        lst_stmt, df_data = U.get_vis(plot_type_map, aDF)
        return lst_stmt, df_data

def get_df_MES(StartDay, EndDay, sql):
    # StartDay = StartDay.strftime('%Y%m%d')
    EndDay = EndDay.strftime('%Y%m%d')
    df_MES = U.get_df_mes(sql.format(StartDay, EndDay))
    df_MES = df_MES.drop_duplicates()
    df_MES['just_date'] = df_MES['WORK_DATE'].str[:4] + '-' + df_MES['WORK_DATE'].str[4:6] + '-' + df_MES['WORK_DATE'].str[6:8]
    return df_MES
def get_df_MES_t(StartDay, EndDay, sql):
    # StartDay = StartDay.strftime('%Y%m%d')
    # EndDay = EndDay.strftime('%Y%m%d')
    df_MES = U.get_df_mes(sql.format(StartDay, EndDay))
    df_MES = df_MES.drop_duplicates()
    df_MES['just_date'] = df_MES['WORK_DATE'].str[:4] + '-' + df_MES['WORK_DATE'].str[4:6] + '-' + df_MES['WORK_DATE'].str[6:8]
    return df_MES
def get_df_PI(StartDay, EndDay):
    df_PI = U.PItag_to_Datframe(p.tag_list_sigMon, StartDay, EndDay, '1m')
    df_PI = df_PI.reset_index().rename(columns={'index': 'Timestamp'})
    df_PI['just_date'] = df_PI['Timestamp'].dt.year.astype(str) + '-' +df_PI['Timestamp'].dt.month.astype(str) + '-' + df_PI['Timestamp'].dt.day.astype(str)
    return df_PI
def get_df_PI_from_2023(StartDay, EndDay):
    df_PI = U.PItag_to_Datframe(p.tag_list_2023, StartDay, EndDay, '1m')
    df_PI = df_PI.reset_index().rename(columns={'index': 'Timestamp'})
    df_PI['just_date'] = df_PI['Timestamp'].dt.year.astype(str) + '-' +df_PI['Timestamp'].dt.month.astype(str) + '-' + df_PI['Timestamp'].dt.day.astype(str)
    return df_PI
def get_df_PI_lst(StartDay, EndDay,tag_list):
    df_PI = U.PItag_to_Datframe(tag_list, StartDay, EndDay, '1m')
    df_PI = df_PI.reset_index().rename(columns={'index': 'Timestamp'})
    df_PI['just_date'] = df_PI['Timestamp'].dt.year.astype(str) + '-' + df_PI['Timestamp'].dt.month.astype(str) + '-' + df_PI['Timestamp'].dt.day.astype(str)
    return df_PI

def get_df_filter(cols,StartDay, EndDay,a_df):
    mask = (a_df['Timestamp'] > StartDay) & (a_df['Timestamp'] <= EndDay)
    cols.append('Timestamp')
    df_r = a_df[cols].loc[mask]
    return df_r

def get_df_Alpur_head(df_PI,df_MES):
    df1 = df_MES[['BATCHNO', 'START_Alpur_head', 'END_Alpur_head']]
    df_PI['Alpur_head_loss'] = df_PI["DC_3 TGH_SPO_LevelLaser1"] - df_PI["DC_3 TGH_PV_LevelLaser2"] - 64
    df2 = df_PI[['Timestamp', 'Alpur_head_loss']]
    df2 = df2.loc[df_PI['Alpur_head_loss'] <= 100]

    df3 = df1.sort_values(by=['BATCHNO']).groupby("BATCHNO").apply(lambda g: df2.loc[
        df2["Timestamp"].between(g["START_Alpur_head"].iloc[0].tz_localize("Asia/Seoul"),
                                 g["END_Alpur_head"].iloc[0].tz_localize("Asia/Seoul")),
        "Alpur_head_loss"].mean()).rename("Alpur_head_loss")
    df3 = df3.reset_index().rename(columns={'index': 'Timestamp'})
    df3['check_Alpur_head_loss'] = np.where(df3['Alpur_head_loss'].between(20, 40), 0, 1)
    df = df_MES.merge(df3, on=['BATCHNO'], how='left')
    df = df.loc[df['BATCHNO']!='S95044'] # PI 시스템 이상으로 제외 2023-08-10
    return df

def get_df_Scale(df_PI,df_MES):
    df1 = df_MES[['BATCHNO', 'START_TIME_EX', 'END_TIME_EX','CA_REMOVE_RATE']]
    df2 = df_PI[['Timestamp', 'Cl_Scale_Usage_Drop', 'Cl_Scale']]
    try :
        df3 = df1.groupby("BATCHNO").apply(lambda g: df2.loc[
            df2["Timestamp"].between(g["START_TIME_EX"].iloc[0].tz_localize("Asia/Seoul"),
                                     g["END_TIME_EX"].iloc[0].tz_localize("Asia/Seoul")),# & df2['Cl_Scale']<300,
            'Cl_Scale_Usage_Drop'].max()).rename('Cl_Scale_Usage_Drop')
        df3 = df3.reset_index().rename(columns={'index': 'Timestamp'})
        df3['check_Cl_Scale_Usage_Drop'] = np.where(df3['Cl_Scale_Usage_Drop'].between(0, 2), 0, 1)
        df = df_MES.merge(df3, on=['BATCHNO'], how='left')
        return df
    except:
        return None

def get_df_chk(df,sig,th):
    # start, end, chk_time 컬럼 초기화
    df['start'] = pd.NaT
    df['end'] = pd.NaT
    df['Period'] = pd.NaT
    df['BATCHNO'] = ''

    # sig 값이 th 이상으로 증가하는 시간을 start로 설정
    mask_start = (df[sig] >= th) & (df[sig].shift() < th)
    df.loc[mask_start, 'start'] = df['Timestamp']

    # sig 값이 th 이하로 감소하는 시간을 end로 설정
    mask_end = (df[sig] <= th) & (df[sig].shift() > th)
    df.loc[mask_end, 'end'] = df['Timestamp']

    df['start'] = df['start'].fillna(method='ffill')
    df['end'] = df['end'].fillna(method='ffill')
    df['Period'] = df['end'] - df['start']
    df = df.loc[(df['end'] > df['start'])&(df['end'] == df['Timestamp'])]

    return df

def get_df_RFI(df_PI,df_MES):
    # x축 : drop No 표기 --> RFI 가동이 안된 경우 Or 가동
    # 30분 이하의 경우도 그래프에 표기 (해당 drop No는 빨간색으로 point)
    df1 = df_MES[['BATCHNO', 'START_TIME_EX', 'END_TIME_EX']]
    df2 = df_PI[['Timestamp', 'New_RFI_Salt_Flow_PV']]
    df2 = get_df_chk(df2, 'New_RFI_Salt_Flow_PV',1.3)

    try:
        df3 = df1.groupby("BATCHNO").apply(lambda g: df2.loc[
            df2["Timestamp"].between(g["START_TIME_EX"].iloc[0].tz_localize("Asia/Seoul"),
                                     g["END_TIME_EX"].iloc[0].tz_localize("Asia/Seoul")),
            'Period'].max()).rename('Period')
        df3 = df3.reset_index().rename(columns={'index': 'Timestamp'})
        df3['chk_time'] = False
        if (~pd.isnull(df3['Period']).all()):
            df3['chk_time'] = (df3['Period'] <= pd.to_timedelta('10 minutes'))|(pd.isnull(df3['Period']))
            df3['Period'] = df3['Period'].dt.total_seconds() // 60
        df = df_MES.merge(df3, on=['BATCHNO'], how='left')
        return df
    except:
        return None
    return  None

def get_df_split_jet(df_PI,df_MES):
    df1 = df_MES[['BATCHNO', 'START_TIME_EX', 'END_TIME_EX']]
    df2_1 = df_PI[['Timestamp', 'DC_3 JET_PV_SplitJetFacePressure']]

    df2 = get_df_chk(df2_1.copy(), 'DC_3 JET_PV_SplitJetFacePressure',2900)

    try:
        df3 = df1.groupby("BATCHNO").apply(lambda g: df2.loc[
            df2["Timestamp"].between(g["START_TIME_EX"].iloc[0].tz_localize("Asia/Seoul"),
                                     g["END_TIME_EX"].iloc[0].tz_localize("Asia/Seoul")),
            'Timestamp'].max()).rename('Timestamp')
        df3 = df3.reset_index().rename(columns={'index': 'Timestamp'})
        df4 = df1.merge(df3, on=['BATCHNO'], how='left')
        df4 = df4[['BATCHNO','Timestamp']]
        df = df2_1.merge(df4, on=['Timestamp'], how='left')
        return df
    except :
        return None

    return  None

def get_df_CA_Scale(df_PI,df_MES):
    df1 = df_MES[['BATCHNO', 'START_TIME_EX', 'END_TIME_EX']]
    df2 = df_PI[['Timestamp', 'Cl_Scale_Usage_Drop', 'Cl_Scale']]
    try :
        df3 = df1.groupby("BATCHNO").apply(lambda g: df2.loc[
            df2["Timestamp"].between(g["START_TIME_EX"].iloc[0].tz_localize("Asia/Seoul"),
                                     g["END_TIME_EX"].iloc[0].tz_localize("Asia/Seoul")),# & df2['Cl_Scale']<300,
            'Cl_Scale_Usage_Drop'].max()).rename('Cl_Scale_Usage_Drop')
        df3 = df3.reset_index().rename(columns={'index': 'Timestamp'})
        df3['check_Cl_Scale_Usage_Drop'] = np.where(df3['Cl_Scale_Usage_Drop'].between(0, 2), 0, 1)
        df = df_MES.merge(df3, on=['BATCHNO'], how='left')
        return df
    except:
        return None

def get_clt_table(df_table):
    sample_list = []
    mean_list = []
    std_list = []
    # df_table= df_table.set_index(df_table.columns[0])
    for i in range(4):
        for _ in range(1000):
            x = df_table.iloc[:, i].sample(30).mean()
            sample_list.append(x)
        mean_list.append(np.mean(sample_list))
        std_list.append(np.std(sample_list))
        sample_list.clear()

    clt_table = pd.DataFrame(zip(mean_list, std_list),
                             index=pd.Index(['Nalco8_Turbidity', 'Nalco5_Conductivity', 'Nalco3_pH', 'Nalco4_ORP']),
                             columns=['mean', 'std'])
    # 3-sigma
    clt_table['Max'] = clt_table['mean'] + 3 * clt_table['std']
    clt_table['Min'] = clt_table['mean'] - 3 * clt_table['std']

    return clt_table

def get_df_CoolingTower(StartDay2, ToDay_0630):
    Cols_Nalco = [
                  'Nalco8_Turbidity',
                  'Nalco5_Conductivity',
                  'Nalco3_pH',
                  'Nalco4_ORP',
                  'Delac_1 Conveyor_Feedrate_PV',
                  'Delac_2 Conveyor_Feedrate_PV']
    # df = get_df_PI(MonthStartDay, ToDay_0630)

    df = U.PItag_to_Datframe(Cols_Nalco, StartDay2, ToDay_0630, '1m') #WeekStart
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[df['Delac_1 Conveyor_Feedrate_PV'] > 10000]
    df = df[df['Delac_2 Conveyor_Feedrate_PV'] > 10000]
    # df = df[df['Nalco8_Turbidity'] > 2]
    df.index = pd.to_datetime(df.index)
    del_cols = ['Delac_1 Conveyor_Feedrate_PV', 'Delac_2 Conveyor_Feedrate_PV']
    df.drop(del_cols, axis=1, inplace=True)

    df_today = df.loc[EndDay2: ToDay_0630]

    clt_table = get_clt_table(df)

    return df_today, clt_table

def get_df_CoolingTower2(StartDay_3, ToDay_0630):
    Cols = ["DC_3 PIT_PV_PitWaterLevel","CT LIA_201_LT",
            "CT PCV_202_SV","CT PT_201",
            "DC_3 WTR_SPO_FaceWaterFlow",
            "DC_3 WTR_PV_FaceWaterFlow",
            "DC_3 WTR_SPO_EndWaterFlow",
            "DC_3 WTR_PV_MoldEndWaterFlow"
            ]
    df = U.PItag_to_Datframe(Cols, StartDay_3, ToDay_0630, '1m')
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index = pd.to_datetime(df.index)
    df = df.reset_index().rename(columns={'index': 'Timestamp'})
    return df
def INS_data(data_df):
    logging.info('INS_DATA')
    try:
        conn = pyodbc.connect(p.Azure_conn_string)
        cursor = conn.cursor()
        sql = """
        IF NOT EXISTS (SELECT * FROM [dbo].[yej_rccs_MTD_BY_BATCH] WHERE [SUMDAY] = ? AND [BATCHNO]=? )
            INSERT INTO [dbo].[yej_rccs_MTD_BY_BATCH]([SUMDAY],[BATCHNO],[CAT],[OUTWGT],[SCRAPWGT],[OFFWGT],[WIPWGT]) values (?,?,?,?,?,?,?)
        """
        for index, row in data_df.iterrows():
            cursor.execute(sql, row['SUMDAY'],row['BATCHNO'], row['SUMDAY'],row['BATCHNO'], row['CAT'], row['OUTWGT'], row['SCRAPWGT'], row['OFFWGT'],row['WIPWGT'])
            conn.commit()
        cursor.close()
        conn.close()
    except Exception as err:
        logging.info({"Error": str(err)})
def get_start(today,days):
    StartDay = today - datetime.timedelta(days=days)
    StartDay2 = StartDay.strftime('%Y-%m-%d') + ' 06:30:00'
    return StartDay2
def get_data_2(lst_Target, aDF):
    if aDF is None:
        return [], None
    else:
        plot_type_map = {key: value for key, value in p.plot_type_map.items() if
                         any(string in key for string in lst_Target)}
        lst_stmt, df_data = U.get_vis(plot_type_map, aDF)
        return lst_stmt, df_data
def register_data_day(lst_signal,Start,End,Items,lst_stmt, df_data):
    df_PI = get_df_PI_lst(lst_signal, Start,End)
    df_PI['shift_Time'] = df_PI['Timestamp']- datetime.timedelta(hours=6.5)
    df_PI['just_date'] = df_PI['shift_Time'].dt.strftime('%y-%m-%d')
    df_PI = df_PI.groupby('just_date').apply(lambda g: g[lst_signal].mean())
    df_PI = df_PI.reset_index()
    df_PI = df_PI.sort_values('just_date')
    lst_stmt_r, df_data_r = get_data_2(Items, df_PI)
    lst_stmt = lst_stmt + lst_stmt_r
    df_data = pd.concat([df_data, df_data_r], axis=0)
    return lst_stmt, df_data
def main():

    today = datetime.date.today()
    print('Start Main() at: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    StartDay = today - datetime.timedelta(days=8)
    EndDay = today - datetime.timedelta(days=1)
    StartDay_3_1 = today - datetime.timedelta(days=3)
    StartDay_1 = today - datetime.timedelta(days=1)
    StartDay_2 = today - datetime.timedelta(days=2)


    MonthStartDay = today - datetime.timedelta(days=31)
    MonthStartDay = MonthStartDay.strftime('%Y-%m-%d') + ' 06:30:00'

    ToDay_0630 = today.strftime('%Y-%m-%d') + ' 06:30:00'

    StartDay2 = StartDay.strftime('%Y-%m-%d') + ' 06:30:00'
    EndDay2 = EndDay.strftime('%Y-%m-%d') + ' 06:30:00'
    StartDay_3 = StartDay_3_1.strftime('%Y-%m-%d') + ' 06:30:00'
    StartDay_2_1 = StartDay_2.strftime('%Y-%m-%d') + ' 06:30:00'
    OneYearAgoDay = today - datetime.timedelta(days=365)
    OneYearAgoDay = OneYearAgoDay.strftime('%Y-%m-%d') + ' 06:30:00'


    str_today = str(datetime.date.today())
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    data_path = str(dir) + folder
    #######################################################################################
    df_MES = get_df_MES(StartDay, EndDay,p.sql_SigMon_MES)
    df_MES_2 = get_df_MES(StartDay_2, EndDay,p.sql_SigMon_MES)
    StartDay_Month = today - datetime.timedelta(days=31)
    df_MES_month = get_df_MES(StartDay_Month, EndDay,p.sql_SigMon_MES)
    df_MES_RFI = get_df_MES(StartDay_2, EndDay, p.sql_SigMon_RFI)
    df_MES_RFI_month = get_df_MES(StartDay_Month, EndDay, p.sql_SigMon_RFI)
    df_PI = get_df_PI(StartDay2, ToDay_0630)
    df_PI_MONTH = get_df_PI(MonthStartDay, ToDay_0630)


    PI_Cols = ["CT Cl2_Storage_Cl2_Pressure","Alpur Cl_Main_Pressure",
               "Alpur CHLORINE.AI.Flow_Rotor_1",
               "Alpur TM.Heater1.Power_Mes",
               "DC_3 ROD_PV_TiBorSpeed",
               "DBF_Pree TC_BoxTemp",
               "CT PCV_202_SV",
               "DC_3 WTR_SPO_FaceWaterFlow",
               'Delac_1 WTCT3',
               'Delac_2 WTCT3',
               'Delac_1 WTCT4',
               'Delac_2 WTCT4',
               'New_RFI_Salt_Flow_PV'
               ]
    avg_values_PI = df_PI[PI_Cols].mean().round(1)
    #######################################################################################
    df_Alpur = get_df_Alpur_head(df_PI, df_MES)
    cols = ['RT_1', 'CT_1', 'BUTTCURL', 'DBF_head_loss', 'Alpur_head_loss', 'CA_REMOVE_RATE']
    avg_values_Alpur = df_Alpur[cols].mean().round(1)
    date_Out_Alpur = get_date_Out(df_Alpur, cols)
    #######################################################################################
    df_Scale = get_df_Scale(df_PI, df_MES_2)
    if df_Scale is None:
        avg_values_Scale = pd.Series({'Cl_Scale_Usage_Drop':[]})
        date_Out_Scale = {'Cl_Scale_Usage_Drop':[]}
    else:
        avg_values_Scale = df_Scale[['Cl_Scale_Usage_Drop']].mean().round(1)
        date_Out_Scale = get_date_Out(df_Scale, ['Cl_Scale_Usage_Drop'])
    #######################################################################################
    df_PI_jet = get_df_filter(['DC_3 JET_PV_SplitJetFacePressure'], EndDay2, ToDay_0630, df_PI)
    df_MES_3 = get_df_MES(StartDay, EndDay, p.sql_SigMon_split_jet)
    df_jet = get_df_split_jet(df_PI_jet, df_MES_3)
    avg_values_jet = pd.Series({'': []})
    date_Out_jet = {'': []}
    #######################################################################################
    df_RFI = get_df_RFI(df_PI, df_MES_RFI)
    avg_values_RFI = pd.Series({'Period': []})
    date_Out_RFI = {'Period': []}

    df_RFI_day1 = get_df_RFI(df_PI_MONTH, df_MES_RFI_month)
    df_RFI_day1['chk_time'] = df_RFI_day1['chk_time'].replace({True: 0, False: 1})

    df_RFI_day = pd.DataFrame()
    df_RFI_day['RFI_CNT'] = df_RFI_day1.groupby("WORK_DATE")['chk_time'].sum().rename('RFI_CNT') ## 문제가 있는 RFI 는 제외
    df_RFI_day['RFI_BATCHNO'] = df_RFI_day1.groupby("WORK_DATE")['BATCHNO'].count().rename('RFI_BATCHNO')
    df_RFI_day['RFI_Day'] = (df_RFI_day['RFI_CNT'] / df_RFI_day['RFI_BATCHNO']) * 100
    df_RFI_day['check_RFI_Day'] = ''

    df_RFI_day = df_RFI_day.reset_index().rename(columns={'WORK_DATE': 'just_date'})
    avg_values_RFI_day = df_RFI_day[['RFI_Day']].mean().round(1)
    date_Out_RFI_day = []
    StartDay_2024 = '2023-01-01 06:30:00'
    # df_RFI_Year2023 = pd.read_csv('./data_const/df_RFI_Year2023.csv')
    df_PI_from_2024 = get_df_PI_from_2023(StartDay_2024, ToDay_0630)
    df_MES_RFI_from_2024 = get_df_MES(StartDay_2024, EndDay, p.sql_SigMon_RFI)

    df_RFI_Month1 = get_df_RFI(df_PI_from_2024, df_MES_RFI_from_2024)
    df_RFI_Month1['chk_time'] = df_RFI_Month1['chk_time'].replace({True: 0, False: 1})
    # df_RFI_Month = df_RFI_Year2023
    df_RFI_Month = pd.DataFrame()
    df_RFI_Month['RFI_CNT'] = df_RFI_Month1.groupby("WORK_MONTH")['chk_time'].sum().rename('RFI_CNT')  ## 문제가 있는 RFI 는 제외
    df_RFI_Month['RFI_BATCHNO'] = df_RFI_Month1.groupby("WORK_MONTH")['BATCHNO'].count().rename('RFI_BATCHNO')
    df_RFI_Month['RFI_Month'] = (df_RFI_Month['RFI_CNT'] / df_RFI_Month['RFI_BATCHNO']) * 100
    df_RFI_Month['check_RFI_Month'] = ''

    # df_RFI_Month.reset_index(drop=True, inplace=True)
    # df_RFI_Year2023.reset_index(drop=True, inplace=True)
    # df_RFI_Month = pd.concat([df_RFI_Year2023,df_RFI_Month], axis=0)
    df_RFI_Month = df_RFI_Month.reset_index().rename(columns={'WORK_MONTH': 'just_MONTH'})
    avg_values_RFI_Month = df_RFI_Month[['RFI_Month']].mean().round(1)

    #######################################################################################

    #######################################################################################
    df_Scale_day = get_df_Scale(df_PI_MONTH, df_MES_month)
    df_Scale_day = df_Scale_day.groupby("WORK_DATE")['Cl_Scale_Usage_Drop'].sum().rename('Cl_Scale_Usage_Drop_Day')
    df_Scale_day = df_Scale_day.reset_index().rename(columns={'WORK_DATE': 'just_date'})
    avg_values_Scale_day = df_Scale_day[['Cl_Scale_Usage_Drop_Day']].mean().round(1)
    date_Out_Scale_day = []
    #######################################################################################

    avg_value_M2 = pd.concat([avg_values_Scale,avg_values_PI,avg_values_Alpur,avg_values_Scale_day,avg_values_RFI,avg_values_RFI_day,avg_values_RFI_Month,avg_values_jet], axis=0)
    # avg_value_M2 = pd.concat([avg_values_Scale, avg_values_PI, avg_values_Alpur, avg_values_Scale_day], axis=0)
    date_Out = date_Out_Alpur.copy()
    date_Out.update(date_Out_Scale)
    date_Out.update(date_Out_RFI)
    date_Out.update(date_Out_jet)
    date_Out.update(date_Out_Scale_day)

    lst_Target_Alpur_Head_Loss = ['Alpur Head Loss', 'DBF Head Loss', '주조 초기 용탕 온도', '초기 냉각수 수온', 'Butt curl수준','Ca 제거효율','Ca 제거효율(DROP)']
    lst_stmt, df_data = get_data(lst_Target_Alpur_Head_Loss, date_Out, avg_value_M2, df_Alpur)
    if df_Scale is None:
        lst_stmt_scale, df_data_scale = [],None
        lst_stmt_cl_ca, df_data_cl_ca = [], None
    else:
        lst_stmt_scale, df_data_scale = get_data(['Alpur 염소 사용량'], date_Out, avg_value_M2, df_Scale)
        lst_stmt_cl_ca, df_data_cl_ca = U.CL_CA_Com(df_Scale, date_Out)

    lst_stmt_jet, df_data_jet = get_data(['Split jet valve 압력 모니터링'], date_Out, avg_value_M2, df_jet)

    lst_stmt_rfi, df_data_rfi = get_data(['RFI 가동률'], date_Out, avg_value_M2,df_RFI)
    lst_stmt_rfi_day, df_data_rfi_day = get_data(['일자별_RFI_가동률'], date_Out, avg_value_M2, df_RFI_day)
    lst_stmt_rfi_month, df_data_rfi_month = get_data(['월별_RFI_가동률'], date_Out, avg_value_M2, df_RFI_Month)

    lst_stmt_scale_day, df_data_scale_day = get_data(['Alpur 염소사용량(일자별)'], date_Out, avg_value_M2, df_Scale_day)

    lst_Target_PI = ['Alpur 염소 저장소 Pressure', 'Alpur 염소 Main Panel Pressure',
                     'decoater_1_3_zone_temp', 'decoater_2_3_zone_temp',
                     'decoater_1_4_zone_temp', 'decoater_2_4_zone_temp'
                     ]
    lst_stmt_PI, df_data_PI = get_data(lst_Target_PI, date_Out, avg_value_M2, df_PI)

    lst_Press_Boiler = ['PIT 301 & PIT 402 & PIT 403']
    Cols_Press_Boiler = [
                  'Boiler Waste heat boiler front pressure PIT301',
                  'Boiler dust collector waste gas front pressure PIT402',
                  'Boiler Manned blower front differential pressure PIT403',
                  ]
    df_Press_Boiler  = get_df_filter(Cols_Press_Boiler, EndDay2, ToDay_0630, df_PI)
    lst_stmt_Press_Boiler, df_data_Press_Boiler = get_data(lst_Press_Boiler, date_Out, avg_value_M2, df_Press_Boiler)
    #######################################################################################
    #######################################################################################
    lst_Target_Press = ['노압 압력 모니터링']
    Cols_Press = [
                  "Delac_1 Bag_Pressure_Transmitter PIT111",
                  "Delac_1 System_Pressure_Valve_Motor_Control_Sig",
                  "Delac_1 System_Pressure_Control_Valve_Feedback",
                  "Delac_2 Bag_Pressure_Transmitter PIT121",
                  "Delac_2 System_Pressure_Valve_Motor_Control_Sig",
                  "Delac_2 System_Pressure_Control_Valve_Feedback",
                  'Delac_1 Bag_Pressure_Transmitter PIT112', 'Delac_2 Bag_Pressure_Transmitter PIT122',
                  'Boiler Waste heat boiler front pressure PIT301'
                  ]
    df_Press = get_df_filter(Cols_Press, EndDay2, ToDay_0630, df_PI)
    path_plot, filename = U.Press_Plot(df_Press)

    lst_stmt_Press = list(zip(['노압 압력 모니터링'], [filename]))
    df_Press = pd.DataFrame(list(zip(["노압 압력 모니터링"], ['노압 압력 모니터링'], [filename])),
                                   columns=["Signal", "Stmt", "Plots"])
    df_Press["Date"] = str(datetime.date.today())
    #######################################################################################
    lst_Target_1 = ['Alpur leak','Casting pit water level','Cooling tower cold pond level','활성탄 투입량 체크',
                    'Main differential pressure 7 Baghouse',]
    Cols_1 = ["Alpur CHLORINE_MAIN.AI.Leak","DC_3 PIT_PV_PitWaterLevel","CT LIA_201_LT","14000_HB AI456",
              "Boiler PIT403-PIT402 _PRESS PIDC402",
              ]
    df_1 = get_df_filter(Cols_1, StartDay_3, ToDay_0630, df_PI)
    lst_stmt_1, df_data_1 = get_data(lst_Target_1, date_Out, avg_value_M2, df_1)
    #######################################################################################
    lst_Target_Bag_House = ['ph값/가성소다/소석회/중탄산']
    Cols_Bag_House = [
        "Boiler AE501_PH_Sensor_new", "Boiler AOH storage tank level LT601", "14000_HB AI448", "Boiler_CARBONATE_TANK_weight"
    ]
    df_Bag_House = get_df_filter(Cols_Bag_House, StartDay_3, ToDay_0630, df_PI)
    path_plot, filename = U.Bag_House(df_Bag_House)

    lst_stmt_Bag_House = list(zip(['Bag House'], [filename]))
    df_Bag_House = pd.DataFrame(list(zip(lst_Target_Bag_House, lst_Target_Bag_House, [filename])),
                                 columns=["Signal", "Stmt", "Plots"])
    df_Bag_House["Date"] = str(datetime.date.today())
    #######################################################################################
    #######################################################################################
    lst_Target_Diff_Press = ['Main differential pressure']
    Cols_Diff_Press = [
        "7000_CB BF3_DPT_DUCT01", "7000_HB BF2_DPT_DUCT01","14000_HB AI352","4000_HB_BF6_01_DPT","Boiler PIT403-PIT402 _PRESS PIDC402"
    ]
    df_Diff_Press = get_df_filter(Cols_Diff_Press, EndDay2, ToDay_0630, df_PI)
    path_plot, filename = U.differentiate_pressor(df_Diff_Press)

    lst_stmt_Diff_Press = list(zip(['Main differential pressure'], [filename]))
    df_Diff_Press = pd.DataFrame(list(zip(lst_Target_Diff_Press, lst_Target_Diff_Press, [filename])),
                            columns=["Signal", "Stmt", "Plots"])
    df_Diff_Press["Date"] = str(datetime.date.today())
    #######################################################################################
    lst_Target_Casting_Water = ['Casting water supply pressure']
    Cols_Casting_Water = [
        "CT PCV_202_SV",
        "CT PT_201"
    ]
    df_Casting_Water = get_df_filter(Cols_Casting_Water, EndDay2, ToDay_0630, df_PI)
    lst_stmt_Casting_Water, df_data_Casting_Water = get_data(lst_Target_Casting_Water, date_Out, avg_value_M2, df_Casting_Water)
    #######################################################################################
    #######################################################################################
    lst_Target_Casting_Water_flow = ['Casting water supply flow']
    Cols_Casting_Water_flow = [
        "DC_3 WTR_SPO_FaceWaterFlow",
        "DC_3 WTR_PV_FaceWaterFlow",
        "DC_3 WTR_SPO_EndWaterFlow",
        "DC_3 WTR_PV_MoldEndWaterFlow"
    ]
    df_Casting_Water_flow = get_df_filter(Cols_Casting_Water_flow, EndDay2, ToDay_0630, df_PI)
    lst_stmt_Casting_Water_flow, df_data_Casting_Water_flow = get_data(lst_Target_Casting_Water_flow, date_Out, avg_value_M2,
                                                             df_Casting_Water_flow)

    lst_Target_Flow_Rotor = ['Alpur 염소 Flow']
    Cols_Flow_Rotor = [
                       "DC_3 B_FurnTiltBackLtch",
                       "DC_3 TGH_SPO_LevelLaser1",
                       "Alpur CHLORINE.AI.Flow_Rotor_1",
                       "Alpur CHLORINE.AI.Flow_Rotor_2",
                       "Alpur CHLORINE.AI.Flow_Rotor_3",
                       "Alpur CHLORINE.AI.Flow_Rotor_4"
                       ]
    df_Flow_Rotor = get_df_filter(Cols_Flow_Rotor, EndDay2, ToDay_0630, df_PI)
    lst_stmt_Flow_Rotor, df_data_Flow_Rotor = get_data(lst_Target_Flow_Rotor, date_Out, avg_value_M2, df_Flow_Rotor)
    #######################################################################################
    lst_Target_Power_Mes = ['Heater Power']
    Cols_Power_Mes = [
                      "Alpur TM.Heater1.Power_Mes",
                      "Alpur TM.Heater2.Power_Mes",
                      "Alpur TM.Heater3.Power_Mes",
                      "Alpur TM.Heater4.Power_Mes",
                      "Alpur DI_Lid_Closed"
                      ]
    df_Power_Mes = get_df_filter(Cols_Power_Mes,EndDay2, ToDay_0630, df_PI)
    lst_stmt_Power_Mes, df_data_Power_Mes = get_data(lst_Target_Power_Mes, date_Out, avg_value_M2, df_Power_Mes)
    #######################################################################################
    lst_Target_TC_BoxTemp = ['DBF 예열']
    Cols_TC_BoxTemp = [
                      "DBF_Pree TC_BoxTemp",
                      "DBF_Pree HMI_CoverTemp"
                      ]

    df_TC_BoxTemp_MONTH = get_df_filter(Cols_TC_BoxTemp, MonthStartDay, ToDay_0630, df_PI_MONTH)
    df_TC_BoxTemp_MONTH['CAT'] = 'MONTH'
    Cols_TC_BoxTemp = [
        "DBF_Pree TC_BoxTemp",
        "DBF_Pree HMI_CoverTemp"
    ]
    df_TC_BoxTemp_TODAY = get_df_filter(Cols_TC_BoxTemp, EndDay2, ToDay_0630, df_PI)
    df_TC_BoxTemp_TODAY['CAT'] = 'TODAY'
    df_TC_BoxTemp = pd.concat([df_TC_BoxTemp_MONTH,df_TC_BoxTemp_TODAY])
    lst_Target_TC_BoxTemp, df_data_TC_BoxTemp = get_data(lst_Target_TC_BoxTemp, date_Out, avg_value_M2, df_TC_BoxTemp)
    #######################################################################################
    lst_Target_Ti_B_rod = ['Ti-B rod 투입 현황']
    Cols_Ti_B_rod = [
                      "DC_3 ROD_PV_TiBorSpeed",
                      "DC_3 ROD_PV_MiscSpeed"
                      ]
    df_Ti_B_rod = get_df_filter(Cols_Ti_B_rod, EndDay2, ToDay_0630, df_PI)
    lst_stmt_Ti_B_rod, df_data_Ti_B_rod = get_data(lst_Target_Ti_B_rod, date_Out, avg_value_M2, df_Ti_B_rod)
    #######################################################################################
    #######################################################################################
    df = get_df_filter(['SH BB_DS_BEARING_TEMP','SH BB_NDS_BEARING_TEMP'], StartDay2, ToDay_0630, df_PI)
    lst_stmt_BB_BEARING, df_data_BB_BEARING = get_data(['Debaler Bearing (DR NDR) 온도 Monitoring'], date_Out, avg_value_M2, df)
    df = get_df_filter(['SH RT_DS_BEARING_TEMP','SH RT_NDS_BEARING_TEMP'], StartDay2, ToDay_0630, df_PI)
    lst_stmt_RT_BEARING, df_data_RT_BEARING = get_data(['1 Shredder (DR NDR) 온도 Monitoring'], date_Out, avg_value_M2, df)
    df = get_df_filter(['SH HD_DS_BEARING_TEMP','SH HD_NDS_BEARING_TEMP'], StartDay2, ToDay_0630, df_PI)
    lst_stmt_HD_BEARING, df_data_HD_BEARING = get_data(['2 Shredder (DR NDR) 온도 Monitoring'], date_Out, avg_value_M2, df)
    df = get_df_filter(['M22_Amp_Out', 'M38_Amp_Out'], StartDay2, ToDay_0630, df_PI)
    lst_stmt_Amp_Out, df_data_Amp_Out = get_data(['M22, M38 Conveyor 전류 Monitoring'], date_Out, avg_value_M2, df)

    df = get_df_filter(['FIKE_BB DE_EIV1','FIKE_BB DE_EIV2','FIKE_BB DE_EIV3'], StartDay2, ToDay_0630, df_PI)
    lst_stmt_FIKE_BB, df_data_FIKE_BB = get_data(['Fike system damper Monitoring_ Debaler'], date_Out, avg_value_M2, df)
    df = get_df_filter(['FIKE_SH1 SH1_EIV1','FIKE_SH1 SH1_EIV2'], StartDay2, ToDay_0630, df_PI)
    lst_stmt_FIKE_SH1, df_data_FIKE_SH1 = get_data(['Fike system damper Monitoring_ #1 Shredder'], date_Out, avg_value_M2, df)
    df = get_df_filter(['FIKE_SH2 SH2_EIV1','FIKE_SH2 SH2_EIV2'], StartDay2, ToDay_0630, df_PI)
    lst_stmt_FIKE_SH2, df_data_FIKE_SH2 = get_data(['Fike system damper Monitoring_ #2 Shredder'], date_Out, avg_value_M2, df)

    df = get_df_filter(['SH BB_AMPS','SH RT_AMPS','SH HD_AMPS'], StartDay2, ToDay_0630, df_PI)
    lst_stmt_AMPS, df_data_AMPS = get_data(['Debaler Shreder 1 2 대한 전류값 Monitoring'], date_Out, avg_value_M2, df)
    #####################################################################################################################################
    df = get_df_filter(['Delac_1 Cyclone_Inlet_Air_Temp_Setpoint', 'Delac_1 Cyclone_Inlet_Temp',
                        'Delac_1 Recirculation_Fan_Inlet_Temp', 'Delac_1 Kiln_Inlet_Temperature',
                        'Delac_1 Recirc_Fan_Motor_Requested_Speed'],
                       StartDay2, ToDay_0630, df_PI)
    lst_stmt_Inlet1, df_data_Inlet1 = U.Kiln_Cyclone_temp_speed(df, 1)

    df = get_df_filter(['Delac_2 Cyclone_Inlet_Air_Temp_Setpoint', 'Delac_2 Cyclone_Inlet_Temp',
                        'Delac_2 Recirculation_Fan_Inlet_Temp', 'Delac_2 Kiln_Inlet_Temperature',
                        'Delac_2 Recirc_Fan_Motor_Requested_Speed'],
                       StartDay2, ToDay_0630, df_PI)
    lst_stmt_Inlet2, df_data_Inlet2 = U.Kiln_Cyclone_temp_speed(df, 2)

    df = get_df_filter(['Delac_1 Diverter_Valve_Motor_Control_Signal', 'Delac_1 Diverter_Valve_Control_Valve_Feedback'],
                       StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Deverter_Damper_1, df_data_Deverter_Damper_1 = get_data(['Diverter Damper SP & PV_1'], date_Out,
                                                                     avg_value_M2, df)
    df = get_df_filter(['Delac_2 Diverter_Valve_Motor_Control_Signal', 'Delac_2 Diverter_Valve_Control_Valve_Feedback'],
                       StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Deverter_Damper_2, df_data_Deverter_Damper_2 = get_data(['Diverter Damper SP & PV_2'], date_Out,
                                                                     avg_value_M2, df)

    df = get_df_filter(['Delac_1 Diverter_Valve_Control_Valve_Feedback', 'Delac_1 Recirc_Fan_Motor_Requested_Speed',
                        'Delac_1 WTCT3', 'Delac_1 WTCT4'],
                       StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_ZoneTemp1, df_data_ZoneTemp1 = U.ZoneTemp(df, 1)

    df = get_df_filter(['Delac_2 Diverter_Valve_Control_Valve_Feedback', 'Delac_2 Recirc_Fan_Motor_Requested_Speed',
                        'Delac_2 WTCT3', 'Delac_2 WTCT4'],
                       StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_ZoneTemp2, df_data_ZoneTemp2 = U.ZoneTemp(df, 2)

    df = get_df_filter(['Delac_1 Kiln_Zone_3_Fault', 'Delac_1 Kiln_Zone_4_Fault', 'Delac_1 Conveyor_Feedrate_PV'],
                       StartDay2, ToDay_0630, df_PI)
    lst_stmt_Zone_TC_Fault_1, df_data_Zone_TC_Fault_1 = U.Zone_TC_Fault(df, 1)
    df = get_df_filter(['Delac_2 Kiln_Zone_3_Fault', 'Delac_2 Kiln_Zone_4_Fault', 'Delac_2 Conveyor_Feedrate_PV'],
                       StartDay2, ToDay_0630, df_PI)
    lst_stmt_Zone_TC_Fault_2, df_data_Zone_TC_Fault_2 = U.Zone_TC_Fault(df, 2)
    df = get_df_filter(
        ['Delac_1 Inlet_Airlock_Fault', 'Delac_1 Inlet_Airlock_Faulted_Lower', 'Delac_1 Inlet_Airlock_Faulted_Upper'],
        StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Inlet_Airlock_Fault_1, df_data_Inlet_Airlock_Fault_1 = get_data(['Inlet_Airlock_Fault_1'], date_Out,
                                                                             avg_value_M2, df)
    df = get_df_filter(
        ['Delac_2 Inlet_Airlock_Fault', 'Delac_2 Inlet_Airlock_Faulted_Lower', 'Delac_2 Inlet_Airlock_Faulted_Upper'],
        StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Inlet_Airlock_Fault_2, df_data_Inlet_Airlock_Fault_2 = get_data(['Inlet_Airlock_Fault_2'], date_Out,
                                                                             avg_value_M2, df)

    df = get_df_filter(
        ['Delac_1 Discharge_Airlock_Faulted', 'Delac_1 Discharge_Airlock_Faulted_Lower',
         'Delac_1 Discharge_Airlock_Faulted_Upper'],
        StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Discharge_Airlock_Faulted_1, df_data_Discharge_Airlock_Faulted_1 = get_data(
        ['Discharge_Airlock_Faulted_1'], date_Out, avg_value_M2, df)
    df = get_df_filter(
        ['Delac_2 Discharge_Airlock_Faulted', 'Delac_2 Discharge_Airlock_Faulted_Lower',
         'Delac_2 Discharge_Airlock_Faulted_Upper'],
        StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Discharge_Airlock_Faulted_2, df_data_Discharge_Airlock_Faulted_2 = get_data(
        ['Discharge_Airlock_Faulted_2'], date_Out, avg_value_M2, df)
    df = get_df_filter(['Delac_1 Cyclone_Airlock_Fault'], StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Cyclone_Airlock_Fault_1, df_data_Cyclone_Airlock_Fault_1 = get_data(['Cyclone_Airlock_Fault_1'], date_Out,
                                                                                 avg_value_M2, df)
    df = get_df_filter(['Delac_2 Cyclone_Airlock_Fault', ], StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Cyclone_Airlock_Fault_2, df_data_Cyclone_Airlock_Fault_2 = get_data(['Cyclone_Airlock_Fault_2'], date_Out,
                                                                                 avg_value_M2, df)

    df = get_df_filter(['Delac_1 Kiln_Debris_Airlock_Fault'], StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Kiln_Debris_Airlock_Fault_1, df_data_Kiln_Debris_Airlock_Fault_1 = get_data(
        ['Kiln_Debris_Airlock_Fault_1'], date_Out, avg_value_M2, df)
    df = get_df_filter(['Delac_2 Kiln_Debris_Airlock_Fault', ], StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Kiln_Debris_Airlock_Fault_2, df_data_Kiln_Debris_Airlock_Fault_2 = get_data(
        ['Kiln_Debris_Airlock_Fault_2'], date_Out, avg_value_M2, df)

    df = get_df_filter(['Delac_1 Conveyor_Feedrate_PV'], StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Conveyor_Feedrate_PV_1, df_data_Conveyor_Feedrate_PV_1 = get_data(
        ['Conveyor_Feedrate_PV_1'], date_Out, avg_value_M2, df)
    df = get_df_filter(['Delac_2 Conveyor_Feedrate_PV', ], StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Conveyor_Feedrate_PV_2, df_data_Conveyor_Feedrate_PV_2 = get_data(
        ['Conveyor_Feedrate_PV_2'], date_Out, avg_value_M2, df)

    df = get_df_filter(
        ['Delac_1 WaterFlow_Afterburner_DayTot', 'Delac_1 WaterFlow_Duct_DayTot', 'Delac_1 WaterFlow_Kiln_DayTot'],
        StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Water_Spray_flow_1, df_data_Water_Spray_flow_1 = get_data(['Water Spray flow_1'], date_Out, avg_value_M2,
                                                                       df)

    df = get_df_filter(
        ['Delac_2 WaterFlow_Afterburner_DayTot', 'Delac_2 WaterFlow_Duct_DayTot', 'Delac_2 WaterFlow_Kiln_DayTot'],
        StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Water_Spray_flow_2, df_data_Water_Spray_flow_2 = get_data(['Water Spray flow_2'], date_Out, avg_value_M2,
                                                                       df)

    df = get_df_filter(['Delac_1 Recirculation_Fan_Inlet_Temp', 'Delac_1 RC_FAN_Vibration'],
                       StartDay2, ToDay_0630, df_PI)
    lst_stmt_RC_FAN_Vibration1, df_data_RC_FAN_Vibration1 = U.RC_FAN_Vibration(df, 1)

    df = get_df_filter(['Delac_2 Recirculation_Fan_Inlet_Temp', 'Delac_2 RC_FAN_Vibration'],
                       StartDay2, ToDay_0630, df_PI)
    lst_stmt_RC_FAN_Vibration2, df_data_RC_FAN_Vibration2 = U.RC_FAN_Vibration(df, 2)

    df = get_df_filter(['Delac_1 RCFan_Pulley_Bearing_Temp', 'Delac_1 RCFan_Fan_Bearing_Temp',
                        'Delac_2 RCFan_Pulley_Bearing_Temp', 'Delac_2 RCFan_Fan_Bearing_Temp'], StartDay_1_1,
                       ToDay_0630, df_PI)
    # lst_stmt_Fan_Bearing_1, df_data_Fan_Bearing_1 = get_data(['RC_Fan_bearing_temp_pulley_fan'], date_Out,
    #                                                          avg_value_M2, df)
    lst_stmt_Fan_Bearing_1, df_data_Fan_Bearing_1 = U.RC_Fan_bearing_temp_pulley_fan(df)


    df = get_df_filter(['Delac_1 Kiln_Drive_Output_Current','Delac_2 Kiln_Drive_Output_Current'], StartDay_1_1,
                       ToDay_0630, df_PI)
    lst_stmt_Kiln_Driving_Motor, df_data_Kiln_Driving_Motor = get_data(['Kiln_Driving_Motor'], date_Out,
                                                                           avg_value_M2, df)

    # df = get_df_filter(['Delac_2 Kiln_Drive_Output_Current'], StartDay_1_1,
    #                    ToDay_0630, df_PI)
    # lst_stmt_Kiln_Driving_Motor_2, df_data_Kiln_Driving_Motor_2 = get_data(['Kiln_Driving_Motor_2'], date_Out,
    #                                                                        avg_value_M2, df)

    df = get_df_filter(['Delac_1 Kiln_Bearing_1_Temp', 'Delac_1 Kiln_Bearing_2_Temp', 'Delac_1 Kiln_Bearing_3_Temp',
                        'Delac_1 Kiln_Bearing_4_Temp',
                        'Delac_1 Kiln_Bearing_5_Temp', 'Delac_1 Kiln_Bearing_6_Temp', 'Delac_1 Kiln_Bearing_7_Temp',
                        'Delac_1 Kiln_Bearing_8_Temp'],
                       StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Support_Roller_Temp_1, df_data_Support_Roller_Temp_1 = U.support_roller_bearing_temp(df,1)
    df = get_df_filter(['Delac_2 Kiln_Bearing_1_Temp', 'Delac_2 Kiln_Bearing_2_Temp', 'Delac_2 Kiln_Bearing_3_Temp',
                        'Delac_2 Kiln_Bearing_4_Temp',
                        'Delac_2 Kiln_Bearing_5_Temp', 'Delac_2 Kiln_Bearing_6_Temp', 'Delac_2 Kiln_Bearing_7_Temp',
                        'Delac_2 Kiln_Bearing_8_Temp'],
                       StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_Support_Roller_Temp_2, df_data_Support_Roller_Temp_2 = U.support_roller_bearing_temp(df,2)

    df = get_df_filter(['Delac_1 Kiln_Inlet_O2', 'Delac_1 Kiln_Discharge_O2', 'Delac_1 Afterburner_O2'],
                       StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_O2_1, df_data_O2_1 = get_data(['O2_lnlet_discharge_afterburner_1'], date_Out, avg_value_M2, df)
    df = get_df_filter(['Delac_2 Kiln_Inlet_O2', 'Delac_2 Kiln_Discharge_O2', 'Delac_2 Afterburner_O2'],
                       StartDay_1_1, ToDay_0630, df_PI)
    lst_stmt_O2_2, df_data_O2_2 = get_data(['O2_lnlet_discharge_afterburner_2'], date_Out, avg_value_M2, df)

    df = get_df_filter(['M24_Amp_Out', 'M40_Amp_Out'], StartDay2, ToDay_0630, df_PI)
    lst_stmt_M24_40_Amp_Out, df_data_M24_40_Amp_Out = get_data(['M24_40_Amp_Out'], date_Out, avg_value_M2, df)
    df = get_df_filter(['M25_Amp_Out'], StartDay2, ToDay_0630, df_PI)
    lst_stmt_M25_Amp_Out, df_data_M25_Amp_Out = get_data(['M25_Amp_Out'], date_Out, avg_value_M2, df)
    #######################################################################################
    lst_Target_CoolingTower = ['Cooling Tower']
    df_today, clt_table = get_df_CoolingTower(StartDay2, ToDay_0630)
    path_plot, filename = U.CoolingTower_Plot(clt_table, df_today)
    lst_stmt_CoolingTower = list(zip(['CoolingTower'], [filename]))
    df_CoolingTower = pd.DataFrame(list(zip(["Cooling Tower"], ['CoolingTower'], [filename])),
                           columns=["Signal", "Stmt", "Plots"])
    df_CoolingTower["Date"] = str(datetime.date.today())
    #################################################################################################
    #######################################################################################
    lst_Target_CoolingTower2 = ['Cooling Tower2']
    df_CoolingTower2 = get_df_CoolingTower2(StartDay_3, ToDay_0630)
    path_plot, filename = U.CoolingTower_Plot2(df_CoolingTower2)
    lst_stmt_CoolingTower2 = list(zip(['CoolingTower2'], [filename]))
    df_CoolingTower2 = pd.DataFrame(list(zip(["Cooling Tower2"], ['CoolingTower2'], [filename])),
                                   columns=["Signal", "Stmt", "Plots"])
    df_CoolingTower2["Date"] = str(datetime.date.today())
    #################################################################################################
    lst_stmt = lst_stmt_scale + lst_stmt_scale_day + lst_stmt_Ti_B_rod + lst_stmt_Flow_Rotor + lst_stmt_Power_Mes + lst_stmt_PI \
               + lst_Target_TC_BoxTemp + lst_stmt + lst_stmt_CoolingTower + lst_stmt_1 + lst_stmt_Casting_Water + lst_stmt_Casting_Water_flow \
               + lst_stmt_Press_Boiler + lst_stmt_Press + lst_stmt_Diff_Press + lst_stmt_Bag_House + lst_stmt_CoolingTower2 + lst_stmt_cl_ca + lst_stmt_rfi + lst_stmt_rfi_day +lst_stmt_rfi_month \
               + lst_stmt_jet + lst_stmt_BB_BEARING + lst_stmt_BB_BEARING + lst_stmt_RT_BEARING + lst_stmt_HD_BEARING + lst_stmt_Amp_Out + lst_stmt_FIKE_BB + lst_stmt_FIKE_SH1 + lst_stmt_FIKE_SH2 + lst_stmt_AMPS \
               + lst_stmt_Inlet1 + lst_stmt_Inlet2 + lst_stmt_Deverter_Damper_1 + lst_stmt_Deverter_Damper_2 + lst_stmt_ZoneTemp1 + lst_stmt_ZoneTemp2 \
               + lst_stmt_Zone_TC_Fault_1 + lst_stmt_Zone_TC_Fault_2 + lst_stmt_Inlet_Airlock_Fault_1 + lst_stmt_Inlet_Airlock_Fault_2 \
    +lst_stmt_Discharge_Airlock_Faulted_1 + lst_stmt_Discharge_Airlock_Faulted_2 + lst_stmt_Cyclone_Airlock_Fault_1 + lst_stmt_Cyclone_Airlock_Fault_2 \
    +lst_stmt_Kiln_Debris_Airlock_Fault_1 + lst_stmt_Kiln_Debris_Airlock_Fault_2 + lst_stmt_Conveyor_Feedrate_PV_1 + lst_stmt_Conveyor_Feedrate_PV_2 \
    +lst_stmt_Water_Spray_flow_1 + lst_stmt_Water_Spray_flow_2 + lst_stmt_RC_FAN_Vibration1 + lst_stmt_RC_FAN_Vibration2 \
    +lst_stmt_Fan_Bearing_1 + lst_stmt_Kiln_Driving_Motor \
    +lst_stmt_Support_Roller_Temp_1 + lst_stmt_Support_Roller_Temp_2 + lst_stmt_O2_1 + lst_stmt_O2_2 + lst_stmt_M24_40_Amp_Out + lst_stmt_M25_Amp_Out

    df_data = pd.concat([df_data_scale, df_data_scale_day, df_data_Ti_B_rod, df_CoolingTower, df_data_Flow_Rotor
                            , df_data_Power_Mes, df_data_TC_BoxTemp, df_data_1, df_data_Casting_Water,
                         df_data_Casting_Water_flow,
                         df_data_Press_Boiler, df_data_PI, df_data, df_Press, df_Diff_Press, df_Bag_House,
                         df_CoolingTower2,
                         df_data_cl_ca, df_data_rfi, df_data_rfi_day,df_data_rfi_month,
                         df_data_jet, df_data_BB_BEARING, df_data_RT_BEARING, df_data_HD_BEARING, df_data_Amp_Out,
                         df_data_FIKE_BB, df_data_FIKE_SH1, df_data_FIKE_SH2, df_data_AMPS
                            , df_data_Inlet1, df_data_Inlet2, df_data_Deverter_Damper_1, df_data_Deverter_Damper_2,
                         df_data_ZoneTemp1, df_data_ZoneTemp2
                            , df_data_Zone_TC_Fault_1, df_data_Zone_TC_Fault_2, df_data_Inlet_Airlock_Fault_1,
                         df_data_Inlet_Airlock_Fault_2
                            , df_data_Discharge_Airlock_Faulted_1, df_data_Discharge_Airlock_Faulted_2,
                         df_data_Cyclone_Airlock_Fault_1, df_data_Cyclone_Airlock_Fault_2
                            , df_data_Kiln_Debris_Airlock_Fault_1, df_data_Kiln_Debris_Airlock_Fault_2,
                         df_data_Conveyor_Feedrate_PV_1, df_data_Conveyor_Feedrate_PV_2
                            , df_data_Water_Spray_flow_1, df_data_Water_Spray_flow_2, df_data_RC_FAN_Vibration1,
                         df_data_RC_FAN_Vibration2
                            , df_data_Fan_Bearing_1, df_data_Kiln_Driving_Motor
                            , df_data_Support_Roller_Temp_1, df_data_Support_Roller_Temp_2, df_data_O2_1, df_data_O2_2,
                         df_data_M24_40_Amp_Out, df_data_M25_Amp_Out
                         ], axis=0)

    # lst_stmt, df_data = register_data_day(['CST_Recycle.ColdLine_ColdLine1_Debaler_E',
    # 'CST_Recycle.ColdLine_ColdLine1_Shredder1_E',
    # 'CST_Recycle.ColdLine_ColdLine1_Shredder2_E'], get_start(today, 30), ToDay_0630, ['모터 전력량 집계'], lst_stmt, df_data)
    # lst_stmt, df_data = register_data_day(['CST_Recycle.Local17_2F_Main_PR_E'], get_start(today, 30),
    #                                   ToDay_0630, ['#17 ECR 전력량 집계'], lst_stmt, df_data)
    # lst_stmt, df_data = register_data_day(['CST_Recycle.ColdLine_Local18_1F_Main_PR_E',
    # 'CST_Recycle.MeltingFurnace_Local19_PR_E',
    # 'CST_Recycle.Casting_Local20_Main_PR_E'], get_start(today, 30),
    #                                   ToDay_0630, ['#18 ~ #20 ECR 전력량 집계'], lst_stmt, df_data)
    # lst_stmt, df_data = register_data_day(['CST_Recycle.ColdLine_Decoater1+2_4BagHouse_E',
    # 'CST_Recycle.MeltingFurnace_SidewellMelter1+2+3_5BagHouse_Inverter_E',
    # 'CST_Recycle.MeltingFurnace_SidewellMelter4_6BagHouse_E'], get_start(today, 30),
    #                                   ToDay_0630, ['#3 ~ #7 Bag house 전력량 집계'], lst_stmt, df_data)

    logger.info("Data generation completed")

    try:
        data_file_path = 'D:/Python/DecoaterFeedRate/data/data_{}.csv'.format(str_today)
        # data_file_path = 'D:/Python/DecoaterFeedRate/data/data_{}.csv'.format(str_today)
        # data_file_path = 'C:/Users/leec/OneDrive - Novelis Inc/Python/Practice/DecoaterFeedRate/data/data_{}.csv'.format(str_today)
        df_data.to_csv(data_file_path, index=False, encoding="utf-8-sig")
        U.send_to_DBF(lst_stmt, data_file_path)
        # U.send_email_seq(df_data)

    except Exception as err:
        print({"Error": str(err)})
        logger.info({"Error": str(err)})
        U.send_email_Error(str(err))
    print('End Main() at: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    pass
    # U.send_to_SharePoint(lst_stmt)
    # recipients = ['changhyuck.lee@novelis.com']
    # U.send_email_multi(recipients, lst_Target, lst_stmt)
def job():
    try:
        # main() 함수 호출
        main()
    except Exception as e:
        # 예외가 발생한 경우, 예외 내용을 출력하고 예외를 무시합니다.
        print("Error occurred in main():", str(e))
        U.send_email_Error(str(e))
def get_df_Scale_total(dates_list,year):
    # 결과 출력
    df_Scale_total = pd.DataFrame()
    for first, next_month_first in dates_list:
        MonthStartDay=first + ' 06:30:00'
        EndDay = next_month_first + ' 06:30:00'
        print('MonthStartDay : ',MonthStartDay)
        df_PI_MONTH = get_df_PI(MonthStartDay, EndDay)
        df_MES_month = get_df_MES_t(MonthStartDay, EndDay, p.sql_SigMon_MES)
        df_Scale_day = get_df_Scale(df_PI_MONTH, df_MES_month)
        df_Scale_day = df_Scale_day.groupby("WORK_DATE")['Cl_Scale_Usage_Drop'].sum().rename('Cl_Scale_Usage_Drop_Day')
        df_Scale_total = pd.concat([df_Scale_total,df_Scale_day])
    df_Scale_total.to_pickle(f'./data/df_Scale_total_{year}.pickle')

if __name__ == "__main__":
    # U.remove_folder()
    # U.retention_file();
    # StartDay = today - datetime.timedelta(days=72)
    # StartDay2 = StartDay.strftime('%Y-%m-%d') + ' 06:30:00'
    # # StartDay_1 = '2023-11-06  06:30:00'
    # StartDay_1_1 = '20240101'
    # # df_PI = get_df_PI(StartDay2, ToDay_0630)
    # df_PI = U.PItag_to_Datframe(['New_RFI_Salt_Flow_PV'], StartDay2, ToDay_0630, '1m')
    # df_PI = df_PI.reset_index().rename(columns={'index': 'Timestamp'})
    # df_PI['just_date'] = df_PI['Timestamp'].dt.month.astype(str) + '-' + df_PI['Timestamp'].dt.day.astype(str)
    # df_MES_RFI = get_df_MES(StartDay_1, EndDay, p.sql_SigMon_RFI)
    # #
    # # df_RFI = get_df_RFI(df_PI, df_MES_RFI)
    # # df_RFI.to_pickle('./data/RFI.pickle')
    # # #
    # df_PI = get_df_PI(StartDay2, ToDay_0630)
    # df_MES_2 = get_df_MES(StartDay_1_1, EndDay,p.sql_SigMon_MES)
    # df_Scale = get_df_Scale(df_PI, df_MES_2)
    # df_Scale.to_pickle('./data/df_Scale.pickle')

    # MonthStartDay = today - datetime.timedelta(days=809)
    # MonthStartDay = MonthStartDay.strftime('%Y-%m-%d') + ' 06:30:00'
    # EndDay = today - datetime.timedelta(days=629)
    # EndDay = EndDay.strftime('%Y-%m-%d') + ' 06:30:00'
    # df_PI_MONTH = get_df_PI(MonthStartDay, EndDay)
    # df_MES_month = get_df_MES(MonthStartDay, EndDay, p.sql_SigMon_MES)
    # df_Scale_day = get_df_Scale(df_PI_MONTH, df_MES_month)
    # df_Scale_day = df_Scale_day.groupby("WORK_DATE")['Cl_Scale_Usage_Drop'].sum().rename('Cl_Scale_Usage_Drop_Day')
    # df_Scale_day.to_pickle('./data/df_Scale_day.pickle')

    # from datetime import datetime, timedelta
    #
    # # 오늘 날짜를 기준으로 설정합니다.
    # today = datetime.now()
    #
    # # 2022년 1월 1일부터 시작합니다.
    # start_date = datetime(2022, 1, 1)
    #
    # # 각 달의 첫 번째 날과 다음 달의 첫 번째 날을 담을 리스트 초기화
    # dates_list = []
    #
    # # start_date부터 오늘까지 각 달에 대해 반복합니다.
    # current_date = start_date
    # while current_date <= today:
    #     # 해당 달의 첫 번째 날
    #     first_day = current_date.replace(day=1)
    #     # 다음 달의 첫 번째 날을 구합니다.
    #     next_month_first_day = first_day.replace(
    #         month=first_day.month % 12 + 1) if first_day.month < 12 else first_day.replace(year=first_day.year + 1,
    #                                                                                        month=1)
    #
    #     # 리스트에 추가합니다.
    #     dates_list.append((first_day.strftime('%Y-%m-%d'), next_month_first_day.strftime('%Y-%m-%d')))
    #
    #     # 다음 달로 넘어갑니다.
    #     current_date = next_month_first_day
    #

    main();
    schedule.every().day.at("06:40").do(job)
    while True:
        schedule.run_pending()
        time.sleep(60)
        print('sleeping: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
