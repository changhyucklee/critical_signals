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

# log 출력 형식
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG,  format=formatter)
# 로그 생성
logger = logging.getLogger()

warnings.filterwarnings("ignore")
# log를 파일에 출력
file_handler = logging.FileHandler('my.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


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
def get_vis(plot_type_map,df_v):
    lst_avg = []
    files = []

    for target in plot_type_map.keys():
        plot_info = plot_type_map[target]
        x_col = plot_info['x_col']
        y_col = plot_info['y_col']
        avg_stmt = target + ' 평균: {}'.format(plot_info['avg'])
        lst_Out = plot_info['date_out']
        plot = plot_info['plot']
        y_min = plot_info['y_min']
        y_max = plot_info['y_max']
        if lst_Out:
            avg_stmt += ', 관리범위 벗어난 기간: {}'.format(lst_Out)
        lst_avg.append(avg_stmt)
        path_plot, filename = U.create_visualization(target,df_v, x_col, y_col, lst_Out, plot,y_min,y_max)
        files.append(filename)
    lst_stmt = list(zip(lst_avg, files))
    df_data = pd.DataFrame(list(zip(plot_type_map.keys(), lst_avg, files)),
                           columns=["Signal", "Stmt", "Plots"])
    df_data["Date"] = str(datetime.date.today())

    return lst_stmt,df_data


def get_plot_type_map(lst_Target,date_Out,avg_values):
    plot_type_map = {
        'Ti-B rod 투입 현황': {'x_col': 'Timestamp', 'y_col': 'DC_3 ROD_PV_TiBorSpeed',
                         'date_out': [],
                         'avg': avg_values['DC_3 ROD_PV_TiBorSpeed'], 'plot': 'line_raw_multi', 'y_min': 30,
                         'y_max': 60},
        '주조 초기 용탕 온도': {'x_col': 'just_date', 'y_col': 'RT_1', 'date_out': date_Out['RT_1'], 'avg': avg_values['RT_1'],
                        'plot': 'box','y_min':682,'y_max':3.5},
        '초기 냉각수 수온': {'x_col': 'just_date', 'y_col': 'CT_1', 'date_out': date_Out['CT_1'], 'avg': avg_values['CT_1'],
                      'plot': 'box','y_min':30,'y_max':3.5},

        'decoater 1 #3zone temp': {'x_col': 'just_date', 'y_col': 'Delac_1 WTCT3',
                                   'date_out': [], 'avg': avg_values['Delac_1 WTCT3'],
                                    'plot': 'box', 'y_min': 430, 'y_max': 3.5},
        'decoater 1 #4zone temp': {'x_col': 'just_date', 'y_col': 'Delac_1 WTCT4',
                                   'date_out':[], 'avg': avg_values['Delac_1 WTCT4'],
                                   'plot': 'box', 'y_min': 530, 'y_max': 3.5},
        'decoater 2 #3zone temp': {'x_col': 'just_date', 'y_col': 'Delac_2 WTCT3',
                                   'date_out': [], 'avg': avg_values['Delac_2 WTCT3'],
                                   'plot': 'box', 'y_min': 430, 'y_max': 3.5},
        'decoater 2 #4zone temp': {'x_col': 'just_date', 'y_col': 'Delac_2 WTCT4',
                                   'date_out':[], 'avg': avg_values['Delac_2 WTCT4'],
                                   'plot': 'box', 'y_min': 530, 'y_max': 3.5},

        'Butt curl수준': {'x_col': 'just_date', 'y_col': 'BUTTCURL', 'date_out': date_Out['BUTTCURL'],
                        'avg': avg_values['BUTTCURL'], 'plot': 'box','y_min':40,'y_max':45},
        'Alpur 염소 사용량': {'x_col': 'BATCHNO', 'y_col': 'Cl_Scale_Usage_Drop', 'date_out': date_Out['Cl_Scale_Usage_Drop'],
                         'avg': avg_values['Cl_Scale_Usage_Drop'], 'plot': 'bar','y_min':0,'y_max':1},
        'Alpur 염소사용량(일자별)': {'x_col': 'just_date', 'y_col': 'Cl_Scale_Usage_Drop_Day',
                         'date_out': [],
                         'avg': avg_values['Cl_Scale_Usage_Drop_Day'], 'plot': 'bar', 'y_min': 0, 'y_max': 10},
        'Alpur Head Loss': {'x_col': 'BATCHNO', 'y_col': 'Alpur_head_loss', 'date_out': date_Out['Alpur_head_loss'],
                            'avg': avg_values['Alpur_head_loss'], 'plot': 'line','y_min':20,'y_max':40},
        'DBF Head Loss': {'x_col': 'BATCHNO', 'y_col': 'DBF_head_loss', 'date_out': date_Out['DBF_head_loss'],
                          'avg': avg_values['DBF_head_loss'], 'plot': 'line','y_min':40,'y_max':60},
        'Ca 제거효율': {'x_col': 'just_date', 'y_col': 'CA_REMOVE_RATE', 'date_out':[],
                          'avg': avg_values['CA_REMOVE_RATE'], 'plot': 'box', 'y_min': 40, 'y_max': 60},
        'Ca 제거효율(DROP)': {'x_col': 'BATCHNO', 'y_col': 'CA_REMOVE_RATE', 'date_out': [],
                          'avg': avg_values['CA_REMOVE_RATE'], 'plot': 'line', 'y_min': 40, 'y_max': 60},
        'Alpur 염소 저장소 Pressure': {'x_col': 'Timestamp', 'y_col': 'CT Cl2_Storage_Cl2_Pressure', 'date_out': [],
                          'avg': avg_values['CT Cl2_Storage_Cl2_Pressure'], 'plot': 'line_raw','y_min':2.5,'y_max':3.5},
        'Alpur 염소 Main Panel Pressure': {'x_col': 'Timestamp', 'y_col': 'Alpur Cl_Main_Pressure',
                                         'date_out': [],
                                         'avg': avg_values['Alpur Cl_Main_Pressure'], 'plot': 'line_raw','y_min':2.5,'y_max':3.5},
        'Alpur 염소 Flow': {'x_col': 'Timestamp', 'y_col': 'Alpur CHLORINE.AI.Flow_Rotor_1',
                                         'date_out': [],
                                         'avg': avg_values['Alpur CHLORINE.AI.Flow_Rotor_1'], 'plot': 'line_raw_multi', 'y_min': 50,
                                         'y_max': 250},
        'Casting water supply pressure': {'x_col': 'Timestamp', 'y_col': 'CT PCV_202_SV',
                          'date_out': [],
                          'avg': avg_values['CT PCV_202_SV'], 'plot': 'line_raw_multi', 'y_min': 0,
                          'y_max': 10},
        'Casting water supply flow': {'x_col': 'Timestamp', 'y_col': 'DC_3 WTR_SPO_FaceWaterFlow',
                                                 'date_out': [],
                                                 'avg': avg_values['DC_3 WTR_SPO_FaceWaterFlow'],
                                                 'plot': 'line_raw_multi', 'y_min': 0,
                                                 'y_max': 1300},
        'PIT 301 & PIT 402 & PIT 403': {'x_col': 'Timestamp', 'y_col': 'Boiler Waste heat boiler front pressure PIT301',
                           'date_out': [],
                           'avg': 0,
                           'plot': 'line_raw_multi', 'y_min': -450,
                           'y_max': 0},
        '#3 Baghouse 차압체크': {'x_col': 'Timestamp', 'y_col': '7000_CB BF3_DPT_DUCT01',
                           'date_out': [],
                           'avg': 0,
                           'plot': 'line_raw', 'y_min': 50,'y_max': 250},
        '#4 Baghouse 차압체크': {'x_col': 'Timestamp', 'y_col': '7000_HB BF2_DPT_DUCT01',
                             'date_out': [],
                             'avg': 0,
                             'plot': 'line_raw', 'y_min': 120, 'y_max': 460},
        '#5 Baghouse 차압체크': {'x_col': 'Timestamp', 'y_col': '14000_HB AI352',
                             'date_out': [],
                             'avg': 0,
                             'plot': 'line_raw', 'y_min': 50, 'y_max': 350},
        '#6 Baghouse 차압체크': {'x_col': 'Timestamp', 'y_col': '4000_HB_BF6_01_DPT',
                             'date_out': [],
                             'avg': 0,
                             'plot': 'line_raw', 'y_min': 150, 'y_max': 320},
        'Main differential pressure #7 Baghouse' : {'x_col': 'Timestamp', 'y_col': 'Boiler PIT403-PIT402 _PRESS PIDC402',
                             'date_out': [],
                             'avg': 0,
                             'plot': 'line_raw', 'y_min': -75, 'y_max': -150},
        'Alpur leak': {'x_col': 'Timestamp', 'y_col': 'Alpur CHLORINE_MAIN.AI.Leak',
                          'date_out': [],
                          'avg': 0, 'plot': 'line_raw', 'y_min': -5, 'y_max': 5},
        'Casting pit water level': {'x_col': 'Timestamp', 'y_col': 'DC_3 PIT_PV_PitWaterLevel',
                       'date_out': [],
                       'avg': 0, 'plot': 'line_raw', 'y_min': 3, 'y_max': 3.5},
        'Cooling tower cold pond level': {'x_col': 'Timestamp', 'y_col': 'CT LIA_201_LT',
                       'date_out': [],
                       'avg': 0, 'plot': 'line_raw', 'y_min': 20, 'y_max': 70},
        '소석회 투입량 체크': {'x_col': 'Timestamp', 'y_col': '14000_HB AI448',
                                    'date_out': [],
                                    'avg': 0, 'plot': 'line_raw', 'y_min': 5000, 'y_max': 7000},
        '활성탄 투입량 체크': {'x_col': 'Timestamp', 'y_col': '14000_HB AI456',
                       'date_out': [],
                       'avg': 0, 'plot': 'line_raw', 'y_min': 3000, 'y_max': 5000},
        '가성소다 탱크 내 잔여량 확인': {'x_col': 'Timestamp', 'y_col': 'Boiler AOH storage tank level LT601',
                       'date_out': [],
                       'avg': 0, 'plot': 'line_raw', 'y_min': 10, 'y_max': 90},
        'pH 값 체크': {'x_col': 'Timestamp', 'y_col': 'Boiler AE501_PH_Sensor_new',
                             'date_out': [],
                             'avg': 0, 'plot': 'line_raw', 'y_min': 8, 'y_max': 10},
        '중탄산 투입량 체크': {'x_col': 'Timestamp', 'y_col': 'Boiler_CARBONATE_TANK_weight',
                    'date_out': [],
                    'avg': 0, 'plot': 'line_raw', 'y_min': 3000, 'y_max': 7000},
        'Heater Power': {'x_col': 'Timestamp', 'y_col': 'Alpur TM.Heater1.Power_Mes',
                          'date_out': [],
                          'avg': avg_values['Alpur TM.Heater1.Power_Mes'], 'plot': 'line_raw_multi', 'y_min': 10,'y_max': 20},
        'DBF 예열': {'x_col': 'Timestamp', 'y_col': 'DBF_Pree TC_BoxTemp',
                         'date_out': [],
                         'avg': avg_values['DBF_Pree TC_BoxTemp'], 'plot': 'line_raw_TC_BoxTemp', 'y_min': 0,'y_max': 700},
        'RFI 가동률': {'x_col': 'BATCHNO', 'y_col': 'Period', 'date_out': date_Out['Period'], 'avg': avg_values['Period'],
                        'plot': 'bar', 'y_min': 0, 'y_max': 20},
        '일자별_RFI_가동률': {'x_col': 'just_date', 'y_col': 'RFI_Day', 'date_out': [], 'avg': avg_values['RFI_Day'],
                        'plot': 'line', 'y_min': 0, 'y_max': 100},
        'Split jet valve 압력 모니터링': {'x_col': 'Timestamp', 'y_col': 'DC_3 JET_PV_SplitJetFacePressure', 'date_out': date_Out[''], 'avg': avg_values[''],
                        'plot': 'line_raw', 'y_min': 2800, 'y_max': 3100},
    }
    # filtered dictionary
    filtered_map = {key: value for key, value in plot_type_map.items() if any(string in key for string in lst_Target)}
    return filtered_map

def get_data(lst_Target,date_Out, avg_values,aDF):
    plot_type_map = get_plot_type_map(lst_Target, date_Out, avg_values)
    lst_stmt, df_data = get_vis(plot_type_map, aDF)
    return lst_stmt, df_data

def get_df_MES(StartDay, EndDay, sql):
    StartDay = StartDay.strftime('%Y%m%d')
    EndDay = EndDay.strftime('%Y%m%d')
    df_MES = U.get_df_mes(sql.format(StartDay, EndDay))
    df_MES = df_MES.drop_duplicates()
    df_MES['just_date'] = df_MES['WORK_DATE'].str[4:6] + '-' + df_MES['WORK_DATE'].str[6:8]
    return df_MES

def get_df_PI(StartDay, EndDay):
    df_PI = U.PItag_to_Datframe(p.tag_list_sigMon, StartDay, EndDay, '1m')
    df_PI = df_PI.reset_index().rename(columns={'index': 'Timestamp'})
    df_PI['just_date'] = df_PI['Timestamp'].dt.month.astype(str) + '-' + df_PI['Timestamp'].dt.day.astype(str)
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

    df3 = df1.groupby("BATCHNO").apply(lambda g: df2.loc[
        df2["Timestamp"].between(g["START_Alpur_head"].iloc[0].tz_localize("Asia/Seoul"),
                                 g["END_Alpur_head"].iloc[0].tz_localize("Asia/Seoul")),
        "Alpur_head_loss"].mean()).rename("Alpur_head_loss")
    df3 = df3.reset_index().rename(columns={'index': 'Timestamp'})
    df3['check_Alpur_head_loss'] = np.where(df3['Alpur_head_loss'].between(20, 40), 0, 1)
    df = df_MES.merge(df3, on=['BATCHNO'], how='left')

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

def get_df_CoolingTower():
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

def get_df_CoolingTower2():
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

def main():
    str_today = str(datetime.date.today())
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    data_path = str(dir) + folder
    #######################################################################################
    df_MES = get_df_MES(StartDay, EndDay,p.sql_SigMon_MES)
    df_MES_2 = get_df_MES(StartDay_1, EndDay,p.sql_SigMon_MES)
    StartDay_Month = today - datetime.timedelta(days=31)
    df_MES_month = get_df_MES(StartDay_Month, EndDay,p.sql_SigMon_MES)
    df_MES_RFI = get_df_MES(StartDay_1, EndDay, p.sql_SigMon_RFI)
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
    if df_Scale.empty:
        avg_values_Scale = pd.Series({'Cl_Scale_Usage_Drop':[]})
        date_Out_Scale = {'Cl_Scale_Usage_Drop':[]}
    else:
        avg_values_Scale = df_Scale[['Cl_Scale_Usage_Drop']].mean().round(1)
        date_Out_Scale = get_date_Out(df_Scale, ['Cl_Scale_Usage_Drop'])
    #######################################################################################
    # start_1 = pd.to_datetime('2023-05-15', format='%Y-%m-%d').strftime('%Y-%m-%d') + ' 06:30:00'
    # end_1   = pd.to_datetime('2023-05-17', format='%Y-%m-%d').strftime('%Y-%m-%d') + ' 06:30:00'
    # df_PI_jet = get_df_filter(['DC_3 JET_PV_SplitJetFacePressure'],start_1, end_1, df_PI)
    # df_MES_3 = get_df_MES(pd.to_datetime('2023-05-15', format='%Y-%m-%d'), pd.to_datetime('2023-05-17', format='%Y-%m-%d'), p.sql_SigMon_split_jet)

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
    #######################################################################################

    #######################################################################################
    df_Scale_day = get_df_Scale(df_PI_MONTH, df_MES_month)
    df_Scale_day = df_Scale_day.groupby("WORK_DATE")['Cl_Scale_Usage_Drop'].sum().rename('Cl_Scale_Usage_Drop_Day')
    df_Scale_day = df_Scale_day.reset_index().rename(columns={'WORK_DATE': 'just_date'})
    avg_values_Scale_day = df_Scale_day[['Cl_Scale_Usage_Drop_Day']].mean().round(1)
    date_Out_Scale_day = []
    #######################################################################################
    avg_value_M2 = pd.concat([avg_values_Scale,avg_values_PI,avg_values_Alpur,avg_values_Scale_day,avg_values_RFI,avg_values_RFI_day,avg_values_jet], axis=0)
    # avg_value_M2 = pd.concat([avg_values_Scale, avg_values_PI, avg_values_Alpur, avg_values_Scale_day], axis=0)
    date_Out = date_Out_Alpur.copy()
    date_Out.update(date_Out_Scale)
    date_Out.update(date_Out_RFI)
    date_Out.update(date_Out_jet)
    # date_Out.update(date_Out_Scale_day)

    lst_Target = ['Alpur Head Loss', 'DBF Head Loss', '주조 초기 용탕 온도', '초기 냉각수 수온', 'Butt curl수준','Ca 제거효율','Ca 제거효율(DROP)']
    lst_stmt, df_data = get_data(lst_Target, date_Out, avg_value_M2, df_Alpur)
    # lst_stmt_scale, df_data_scale = get_data(['Alpur 염소 사용량','Alpur 염소 사용량(일자별)'], date_Out, avg_value_M2, df_Scale)
    if df_Scale.empty:
        lst_stmt_scale, df_data_scale = [],None
    else:
        lst_stmt_scale, df_data_scale = get_data(['Alpur 염소 사용량'], date_Out, avg_value_M2, df_Scale)

    lst_stmt_jet, df_data_jet = get_data(['Split jet valve 압력 모니터링'], date_Out, avg_value_M2, df_jet)

    lst_stmt_rfi, df_data_rfi = get_data(['RFI 가동률'], date_Out, avg_value_M2,df_RFI)
    lst_stmt_rfi_day, df_data_rfi_day = get_data(['일자별_RFI_가동률'], date_Out, avg_value_M2, df_RFI_day)

    lst_stmt_scale_day, df_data_scale_day = get_data(['Alpur 염소사용량(일자별)'], date_Out, avg_value_M2, df_Scale_day)
    lst_stmt_cl_ca, df_data_cl_ca = U.CL_CA_Com(df_Scale, date_Out)

    lst_Target_PI = ['Alpur 염소 저장소 Pressure', 'Alpur 염소 Main Panel Pressure',
                     'decoater 1 #3zone temp', 'decoater 2 #3zone temp',
                     'decoater 1 #4zone temp', 'decoater 2 #4zone temp'
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
                  "Delac_2 System_Pressure_Control_Valve_Feedback"
                  ]
    df_Press = get_df_filter(Cols_Press, EndDay2, ToDay_0630, df_PI)
    path_plot, filename = U.Press_Plot(df_Press)

    lst_stmt_Press = list(zip(['노압 압력 모니터링'], [filename]))
    df_Press = pd.DataFrame(list(zip(["노압 압력 모니터링"], ['노압 압력 모니터링'], [filename])),
                                   columns=["Signal", "Stmt", "Plots"])
    df_Press["Date"] = str(datetime.date.today())
    #######################################################################################
    lst_Target_1 = ['Alpur leak','Casting pit water level','Cooling tower cold pond level','활성탄 투입량 체크',
                    'Main differential pressure #7 Baghouse',]
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
    lst_Target_CoolingTower = ['Cooling Tower']
    df_today, clt_table = get_df_CoolingTower()
    path_plot, filename = U.CoolingTower_Plot(clt_table, df_today)
    lst_stmt_CoolingTower = list(zip(['CoolingTower'], [filename]))
    df_CoolingTower = pd.DataFrame(list(zip(["Cooling Tower"], ['CoolingTower'], [filename])),
                           columns=["Signal", "Stmt", "Plots"])
    df_CoolingTower["Date"] = str(datetime.date.today())
    #################################################################################################
    #######################################################################################
    lst_Target_CoolingTower2 = ['Cooling Tower2']
    df_CoolingTower2 = get_df_CoolingTower2()
    path_plot, filename = U.CoolingTower_Plot2(df_CoolingTower2)
    lst_stmt_CoolingTower2 = list(zip(['CoolingTower2'], [filename]))
    df_CoolingTower2 = pd.DataFrame(list(zip(["Cooling Tower2"], ['CoolingTower2'], [filename])),
                                   columns=["Signal", "Stmt", "Plots"])
    df_CoolingTower2["Date"] = str(datetime.date.today())
    #################################################################################################
    lst_stmt = lst_stmt_scale+lst_stmt_scale_day+lst_stmt_Ti_B_rod+lst_stmt_Flow_Rotor+ lst_stmt_Power_Mes+ lst_stmt_PI \
               + lst_Target_TC_BoxTemp+ lst_stmt + lst_stmt_CoolingTower+lst_stmt_1+lst_stmt_Casting_Water+lst_stmt_Casting_Water_flow \
               +lst_stmt_Press_Boiler+lst_stmt_Press+lst_stmt_Diff_Press+lst_stmt_Bag_House+lst_stmt_CoolingTower2+lst_stmt_cl_ca+lst_stmt_rfi+lst_stmt_rfi_day \
               + lst_stmt_jet

    df_data = pd.concat([df_data_scale,df_data_scale_day,df_data_Ti_B_rod, df_CoolingTower,df_data_Flow_Rotor
                            ,df_data_Power_Mes, df_data_TC_BoxTemp,df_data_1,df_data_Casting_Water,df_data_Casting_Water_flow,
                         df_data_Press_Boiler,df_data_PI, df_data,df_Press,df_Diff_Press,df_Bag_House,df_CoolingTower2,df_data_cl_ca,df_data_rfi,df_data_rfi_day,
                         df_data_jet],axis=0)
    logger.info("Data generation completed")

    try:
        data_file_path = 'D:/Python/DecoaterFeedRate/data/data_{}.csv'.format(str_today)
        df_data.to_csv(data_file_path, index=False, encoding="utf-8-sig")
        U.send_to_DBF(lst_stmt, data_file_path)
    except Exception as err:
        print({"Error": str(err)})
        logger.info({"Error": str(err)})

    # U.send_to_SharePoint(lst_stmt)
    # recipients = ['changhyuck.lee@novelis.com']
    # U.send_email_multi(recipients, lst_Target, lst_stmt)

if __name__ == "__main__":
    # U.remove_folder()
    # U.retention_file();

    # StartDay2 = '2023-05-01 06:30:00'
    # StartDay_1 = '20230501'
    # df_PI = get_df_PI(StartDay2, ToDay_0630)
    # df_MES_2 = get_df_MES(StartDay_1, EndDay, p.sql_SigMon_MES)
    #
    # df_Scale = get_df_Scale(df_PI, df_MES_2)
    # df_Scale.to_pickle('./data/scale.pickle')
    main();

