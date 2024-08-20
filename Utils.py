from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import warnings

import PIconnect as PI

import pyodbc
from misc import parameters as p

import subprocess
# from sharepoint import SharePoint
from pymongo import MongoClient

import io
import smtplib
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart

from email.mime.image import MIMEImage
from email.message import EmailMessage

import logging

# log 출력 형식
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG,
                    format=formatter)
# 로그 생성
logger = logging.getLogger()

warnings.filterwarnings("ignore")
# log를 파일에 출력
file_handler = logging.FileHandler('my.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def get_df_mes(sql):
    conn = pyodbc.connect(p.conn_string)
    df_input = pd.read_sql(sql, conn)
    conn.close()
    return df_input

def get_df_da(sql):
    pyodbc.autocommit = True
    conn = pyodbc.connect(p.da_conn_string,autocommit=True)
    cursor = conn.cursor()
    df_input = pd.read_sql(sql, conn)
    conn.close()
    return df_input

def find_server():
    PI.PIConfig.DEFAULT_TIMEZONE = 'Asia/Seoul'

    with PI.PIServer() as server:
        return server.server_name
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

def get_filtered_IQR(df,y_col):
    Q1 = df[y_col].quantile(0.25)
    Q3 = df[y_col].quantile(0.75)
    IQR = Q3 - Q1  # IQR is interquartile range.
    filter = (df[y_col] >= Q1 - 1.5 * IQR) & (df[y_col] <= Q3 + 1.5 * IQR)
    df = df.loc[filter]
    return df
def Decoater_temp_Plot(df,cols):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    df_month = df.loc[df.CAT == 'MONTH']

    df_month.drop('CAT', axis=1, inplace=True)
    df_today = df.loc[df.CAT == 'TODAY']
    df_today.drop('CAT', axis=1, inplace=True)
    # df_today['just_date'] = df_today['Timestamp'].dt.month.astype(str) + '-' + df_today['Timestamp'].dt.day.astype(str)

    df = get_filtered_IQR(df,cols[0])
    y_col = df[cols[0]]

    # Month
    plt.subplot(211)
    plt.title('Month', fontsize=15, pad=10)
    x_col = df_month['year_month']
    box1 = sns.boxplot(x=x_col, y=y_col, data=df_month)
    plt.legend()
    # Today
    plt.subplot(212)
    x_col = df_today['just_date']
    plt.title('Today', fontsize=15, pad=10)
    box2 = sns.boxplot(x=x_col, y=y_col, data=df_today)
    plt.legend()

    filename = str(date.today()) + "_Decoater_temp.png"
    # working directory
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    path_plot = str(dir) + folder + filename
    # save plot
    fig.savefig(path_plot, dpi=fig.dpi)

    return path_plot, filename

def differentiate_pressor(df):
    fig, ax = plt.subplots(3, 2, figsize=(30, 15))
    plt.rcParams['axes.unicode_minus'] = False
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    plt.subplot(321)
    plt.title('#3 Baghouse Main differential pressure')
    plt.plot(df['Timestamp'], df['7000_CB BF3_DPT_DUCT01'])
    plt.axhline(y=50, color='red', linestyle='dashed')
    plt.axhline(y=250, color='red', linestyle='dashed')
    plt.subplot(322)
    plt.title('#4 Baghouse Main differential pressure')
    plt.plot(df['Timestamp'], df['7000_HB BF2_DPT_DUCT01'])
    plt.axhline(y=120, color='red', linestyle='dashed')
    plt.axhline(y=460, color='red', linestyle='dashed')
    plt.subplot(323)
    plt.title('#5 Baghouse Main differential pressure')
    plt.plot(df['Timestamp'], df['14000_HB AI352'])
    plt.axhline(y=50, color='red', linestyle='dashed')
    plt.axhline(y=350, color='red', linestyle='dashed')
    plt.subplot(324)
    plt.title('#6 Baghouse Main differential pressure')
    plt.plot(df['Timestamp'], df['4000_HB_BF6_01_DPT'])
    plt.axhline(y=150, color='red', linestyle='dashed')
    plt.axhline(y=320, color='red', linestyle='dashed')
    plt.subplot(325)
    plt.title('#7 Baghouse Main differential pressure')
    plt.plot(df['Timestamp'], df['Boiler PIT403-PIT402 _PRESS PIDC402'])
    plt.axhline(y=-75, color='red', linestyle='dashed')
    plt.axhline(y=-150, color='red', linestyle='dashed')
    filename = str(date.today()) + "_Diff_PPress.png"
    # working directory
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    path_plot = str(dir) + folder + filename
    # save plot
    fig.savefig(path_plot, dpi=fig.dpi)
    return path_plot, filename


def Bag_House(df):
    fig, ax = plt.subplots(2, 2, figsize=(30, 15))
    plt.rcParams['axes.unicode_minus'] = False
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    plt.subplot(221)
    plt.title('pH 값 체크')
    plt.plot(df['Timestamp'], df['Boiler AE501_PH_Sensor_new'])
    plt.axhline(y=8, color='red', linestyle='dashed')
    plt.axhline(y=10, color='red', linestyle='dashed')
    plt.subplot(222)
    plt.title('가성소다 탱크 내 잔여량 확인')
    plt.plot(df['Timestamp'], df['Boiler AOH storage tank level LT601'])
    plt.axhline(y=10, color='red', linestyle='dashed')
    plt.axhline(y=90, color='red', linestyle='dashed')
    plt.subplot(223)
    plt.title('소석회 투입량 체크')
    plt.plot(df['Timestamp'], df['14000_HB AI448'])
    plt.axhline(y=5000, color='red', linestyle='dashed')
    plt.axhline(y=7000, color='red', linestyle='dashed')
    plt.subplot(224)
    plt.title('중탄산 투입량 체크')
    plt.plot(df['Timestamp'], df['Boiler_CARBONATE_TANK_weight'])
    plt.axhline(y=3000, color='red', linestyle='dashed')
    plt.axhline(y=7000, color='red', linestyle='dashed')

    filename = str(date.today()) + "_Bag_House.png"
    # working directory
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    path_plot = str(dir) + folder + filename
    # save plot
    fig.savefig(path_plot, dpi=fig.dpi)
    return path_plot, filename

# https://medium.com/swlh/quick-guide-to-labelling-data-for-common-seaborn-plots-736e10bf14a9
def CoolingTower_Plot(clt_table,df_today):
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels

    plt.subplot(221)
    plt.title('Nalco8_Turbidity', fontsize=15)
    plt.axhline(y=clt_table['Max'][0], color='red', linestyle='dashed')
    plt.axhline(y=clt_table['Min'][0], color='red', linestyle='dashed')
    # plt.axhline(y=10, color='red', linestyle='dashed')
    # plt.axhline(y=0, color='red', linestyle='dashed')
    df_today_1 = df_today[df_today['Nalco8_Turbidity'] > 0]
    plt.plot(df_today_1['Nalco8_Turbidity'], label='Nalco8_Turbidity')
    plt.legend()

    plt.subplot(222)
    plt.title('Nalco5_Conductivity', fontsize=15)
    plt.axhline(y=clt_table['Max'][1], color='red', linestyle='dashed')
    plt.axhline(y=clt_table['Min'][1], color='red', linestyle='dashed')
    # plt.axhline(y=800, color='red', linestyle='dashed')
    # plt.axhline(y=1300, color='red', linestyle='dashed')
    plt.plot(df_today['Nalco5_Conductivity'], label='Nalco5_Conductivity')
    plt.legend()

    plt.subplot(223)
    plt.title('Nalco3_pH', fontsize=15)
    plt.axhline(y=8.8, color='red', linestyle='dashed')
    plt.axhline(y=9.2, color='red', linestyle='dashed')
    plt.plot(df_today['Nalco3_pH'], label='Nalco3_pH')
    plt.legend()

    plt.subplot(224)
    plt.title('Nalco4_ORP', fontsize=15)
    plt.axhline(y=clt_table['Max'][3], color='red', linestyle='dashed')
    plt.axhline(y=clt_table['Min'][3], color='red', linestyle='dashed')
    # plt.axhline(y=100, color='red', linestyle='dashed')
    # plt.axhline(y=500, color='red', linestyle='dashed')
    plt.plot(df_today['Nalco4_ORP'], label='Nalco4_ORP')
    plt.legend()

    filename = str(date.today()) + "_CoolingTower.png"
    # working directory
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    path_plot = str(dir) + folder + filename
    # save plot
    fig.savefig(path_plot, dpi=fig.dpi)
    return path_plot, filename
def CoolingTower_Plot2(df):
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels

    Cols_Flow = ['Timestamp',
            "DC_3 WTR_SPO_FaceWaterFlow",
            "DC_3 WTR_PV_FaceWaterFlow",
            "DC_3 WTR_SPO_EndWaterFlow",
            "DC_3 WTR_PV_MoldEndWaterFlow"
            ]

    plt.subplot(411)
    plt.title('DC_3 PIT_PV_PitWaterLevel', fontsize=15)
    plt.plot(df['DC_3 PIT_PV_PitWaterLevel'], label='DC_3 PIT_PV_PitWaterLevel')
    plt.legend()
    plt.subplot(412)
    plt.title('Pond Level', fontsize=15)
    plt.plot(df['CT LIA_201_LT'], label='CT LIA_201_LTl')
    plt.legend()
    plt.subplot(413)
    plt.title('Casting water supply pressure', fontsize=15)
    line = sns.lineplot(x='Timestamp', y='value', hue='variable', data=pd.melt(df[['Timestamp',"CT PCV_202_SV", "CT PT_201"]], 'Timestamp'))
    ax.set(ylim=(0, 10))
    ax.grid(True, linestyle='--', axis='y')
    plt.legend()
    plt.subplot(414)
    plt.title('Casting water supply flow', fontsize=15)
    line = sns.lineplot(x='Timestamp', y='value', hue='variable', data=pd.melt(df[Cols_Flow], 'Timestamp'))
    ax.grid(True, linestyle='--', axis='y')
    plt.legend()

    filename = str(date.today()) + "_CoolingTower2.png"
    # working directory
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    path_plot = str(dir) + folder + filename
    # save plot
    fig.savefig(path_plot, dpi=fig.dpi)
    return path_plot, filename

def Press_Plot(df):
    import matplotlib.dates as md
    fig, ax = plt.subplots(1,3,figsize=(30, 30))

    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    plt.subplot(311)
    plt.title('Decoater 1')
    plt.plot(df['Timestamp'],df['Delac_1 Bag_Pressure_Transmitter PIT111'], label='Bag_Pressure_Transmitter PIT111', color='red')
    plt.plot(df['Timestamp'],df['Delac_1 System_Pressure_Valve_Motor_Control_Sig'], label='System_Pressure_Valve_Motor_Control_Sig', color='blue')
    plt.plot(df['Timestamp'],df['Delac_1 System_Pressure_Control_Valve_Feedback'], label='System_Pressure_Control_Valve_Feedback')
    plt.plot(df['Timestamp'], df['Delac_1 Bag_Pressure_Transmitter PIT112'], label='Delac_1 Bag_Pressure_Transmitter PIT112', color='black')
    plt.subplot(311).xaxis.set_major_locator(md.MinuteLocator(byminute=[0, 60]))
    # plt.subplot(311).yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    plt.subplot(311).set_ylim([-150, 100])
    plt.tick_params(axis='x', rotation=45, labelsize=15)
    plt.legend()
    plt.subplot(312)
    plt.title('Decoater 2')
    plt.plot(df['Timestamp'],df['Delac_2 Bag_Pressure_Transmitter PIT121'], label='Bag_Pressure_Transmitter PIT121', color='red')
    plt.plot(df['Timestamp'],df['Delac_2 System_Pressure_Valve_Motor_Control_Sig'],
             label='System_Pressure_Valve_Motor_Control_Sig', color='blue')
    plt.plot(df['Timestamp'],df['Delac_2 System_Pressure_Control_Valve_Feedback'],
             label='System_Pressure_Control_Valve_Feedback')
    plt.plot(df['Timestamp'], df['Delac_2 Bag_Pressure_Transmitter PIT122'],
             label='Delac_2 Bag_Pressure_Transmitter PIT122', color='black')
    plt.subplot(312).xaxis.set_major_locator(md.MinuteLocator(byminute=[0, 60]))
    # plt.subplot(312).yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    plt.subplot(312).set_ylim([-150, 100])
    plt.tick_params(axis='x', rotation=45, labelsize=15)
    plt.legend()
    plt.subplot(313)
    plt.title('Decoater1 PIT112 & Decoater2 PIT122')
    plt.plot(df['Timestamp'], df['Delac_1 Bag_Pressure_Transmitter PIT112'],linestyle='dashed', label='Delac_1 Bag_Pressure_Transmitter PIT112', color='black')
    plt.plot(df['Timestamp'], df['Delac_2 Bag_Pressure_Transmitter PIT122'],linestyle='dashed', label='Delac_2 Bag_Pressure_Transmitter PIT122', color='blue')
    plt.plot(df['Timestamp'], df['Boiler Waste heat boiler front pressure PIT301'], linestyle='dashed',
             label='Delac_2 Bag_Pressure_Transmitter PIT122', color='red')
    plt.subplot(313).xaxis.set_major_locator(md.MinuteLocator(byminute=[0, 60]))
    # plt.subplot(313).yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    plt.subplot(313).set_ylim([-200, 50])
    plt.tick_params(axis='x', rotation=45, labelsize=15)
    plt.legend()
    # # y축의 scale을 연한 회색 점선으로 표시하기
    # ax[0].grid(True,axis='y', color='lightgray', linestyle='--')
    # ax[1].grid(True,axis='y', color='lightgray', linestyle='--')
    # for a in ax:
    #     a.set_ylim([-20, 100])
    #     a.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

    filename = str(date.today()) + "_Press.png"
    # working directory
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    path_plot = str(dir) + folder + filename
    # save plot
    fig.savefig(path_plot, dpi=fig.dpi)
    return path_plot, filename
def get_df_PI_lst(tag_list,StartDay, EndDay):
    df_PI_c = PItag_to_Datframe(tag_list, StartDay, EndDay, '1m')
    # df_PI_c2 = U.PItag_to_Datframe(p.tag_list_sigMon, StartDay, EndDay, '1m')
    df_PI_c = df_PI_c.reset_index().rename(columns={'index': 'Timestamp'})
    # df_PI['just_date'] = df_PI['Timestamp'].dt.month.astype(str) + '-' + df_PI['Timestamp'].dt.day.astype(str)
    return df_PI_c
def Main_Fan_Bearing(lst_signal,Start,End,Items,lst_stmt, df_data):
    target = Items
    df = get_df_PI_lst(lst_signal, Start,End)
    import matplotlib.dates as md
    fig, ax = plt.subplots(2, 1, figsize=(30, 30))

    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    plt.subplot(211)
    plt.title('No.5 Baghouse Main Fan Bearing temp')
    plt.plot(df['Timestamp'], df['14000_HB AI320'], label='14000_HB AI320(부하)',color='blue')
    plt.plot(df['Timestamp'], df['14000_HB AI312'], label='14000_HB AI312(반부하)', color='orange')
    plt.legend()
    plt.subplot(212)
    plt.title('No.6 Baghouse Main Fan Bearing temp1')
    plt.plot(df['Timestamp'], df['4000_HB_BF6_FN1_TZ16'], label='4000_HB_BF6_FN1_TZ16(부하)',color='blue')
    plt.plot(df['Timestamp'], df['4000_HB_BF6_FN1_TZ15'],  label='4000_HB_BF6_FN1_TZ15(반부하)', color='orange')
    plt.legend()
    filename = str(date.today()) + "_Main_Fan_Bearing_온도_Monitoring.png"
    # working directory
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    path_plot = str(dir) + folder + filename
    # save plot
    fig.savefig(path_plot, dpi=fig.dpi)

    lst_stmt_r = list(zip([target], [filename]))
    df_data_r = pd.DataFrame(list(zip([target], [target], [filename])),
                      columns=["Signal", "Stmt", "Plots"])
    df_data_r["Date"] = str(date.today())

    lst_stmt = lst_stmt + lst_stmt_r
    df_data = pd.concat([df_data, df_data_r], axis=0)

    return lst_stmt, df_data


def CL_CA_Com(df,lst_Out):
    target ='Alpur 염소 사용량 & Ca 제거 효율(Drop)'
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    if len(lst_Out) > 0:
        pal = {just_date: "r" if just_date in lst_Out else "b" for just_date in df.just_date.unique()}
    else:
        pal = sns.color_palette("mako_r", 6)
    x_col = 'BATCHNO'

    bar = sns.barplot(x=x_col, y='Cl_Scale_Usage_Drop', data=df, ax=ax, ci=None)
    bar.axhline(y=1, color='red', linestyle='dashed')
    bar.bar_label(bar.containers[0], fmt='%.1f')
    ax2 = ax.twinx()
    line = sns.lineplot(x=x_col, y='CA_REMOVE_RATE', data=df, ax=ax2, marker='o', palette=pal)

    ax.tick_params(axis='x', rotation=90)
    ax.grid(True, linestyle='--', axis='y')
    line.set(title=target)

    # label points on the plot
    for x, y, c in zip(df['BATCHNO'], df['CA_REMOVE_RATE'], df['check_CA_REMOVE_RATE']):
        if c == 1:
            aColor = 'red'
        else:
            aColor = 'purple'
        plt.text(x=x, y=y, s='{:.1f}'.format(y), color=aColor)

    filename = str(date.today()) + "_CL_CA_Com.png"
    # working directory
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    path_plot = str(dir) + folder + filename
    # save plot
    fig.savefig(path_plot, dpi=fig.dpi)

    lst_stmt = list(zip([target], [filename]))
    df = pd.DataFrame(list(zip([target], [target], [filename])),
                            columns=["Signal", "Stmt", "Plots"])
    df["Date"] = str(date.today())

    return lst_stmt, df
def save_fig(fig,target):
    filename = str(date.today()) + "_" + target + ".png"
    # working directory
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    path_plot = str(dir) + folder + filename
    # save plot
    fig.savefig(path_plot, dpi=fig.dpi)

    lst_stmt = list(zip([target], [filename]))
    df = pd.DataFrame(list(zip([target], [target], [filename])),
                      columns=["Signal", "Stmt", "Plots"])
    df["Date"] = str(date.today())

    return lst_stmt, df

def Kiln_Cyclone_temp_speed(df,i):
    target ='Kiln Cyclone Inlet temp SP_PV_RC_Fan_Speed_'+str(i)
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels

    df = df.set_index('Timestamp')
    ax = df['Delac_' + str(i) + ' Cyclone_Inlet_Air_Temp_Setpoint'].rolling(window=30).mean().plot(linestyle='--',label='Cyclone_Inlet_Air_Temp_Setpoint')
    ax = df['Delac_' + str(i) + ' Cyclone_Inlet_Temp'].rolling(window=30).mean().plot(label='Cyclone_Inlet_Temp')
    ax = df['Delac_' + str(i) + ' Kiln_Inlet_Temperature'].rolling(window=30).mean().plot(label='Kiln_Inlet_Temperature')
    ax = df['Delac_' + str(i) + ' Recirculation_Fan_Inlet_Temp'].rolling(window=30).mean().plot(color='red', linestyle='--', label='Fan_Inlet_Temp')
    ax.legend(loc='lower left')
    ax2 = ax.twinx()
    ax2 = df['Delac_' + str(i) + ' Recirc_Fan_Motor_Requested_Speed'].rolling(window=30).mean().plot(color='blue', linestyle='--', label='Recirc_Fan_Motor_Requested_Speed')
    ax.set_ylabel('Temp')
    ax2.set_ylabel('Speed')
    ax2.legend(loc='lower right')
    # plt.legend(loc='best')
    ax.set_ylim(200, 450)
    ax2.set_ylim(30, 100)
    ax.set(title=target + ' Trend')

    lst_stmt, df = save_fig(fig, target.replace(' ','_'))

    return lst_stmt, df
def Zone_TC_Fault(df,i):
    target ='Zone_TC_Fault_'+str(i)
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels

    df = df.set_index('Timestamp')
    ax = df['Delac_' + str(i) + ' Kiln_Zone_3_Fault'].plot(color='red',label='Kiln_Zone_3_Fault')
    ax = df['Delac_' + str(i) + ' Kiln_Zone_4_Fault'].plot(color='blue',label='Kiln_Zone_4_Fault')
    ax.legend(loc='lower left')
    ax2 = ax.twinx()
    ax2 = df['Delac_' + str(i) + ' Conveyor_Feedrate_PV'].plot(color='green', linestyle='--', label='Conveyor_Feedrate_PV')
    ax2.legend(loc='lower right')
    # plt.legend(loc='best')
    # ax.set_ylim(200, 450)
    # ax2.set_ylim(30, 100)
    ax.set(title=target + ' Trend')

    lst_stmt, df = save_fig(fig, target)

    return lst_stmt, df


def ZoneTemp(df,i):
    target ='Zone Temp and RC FAN and DIVERTER_'+str(i)
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    df = df.set_index('Timestamp')

    ax = df['Delac_' + str(i) + ' WTCT3'].rolling(window=10).mean().plot(color='red',label='WTCT3')
    ax = df['Delac_' + str(i) + ' WTCT4'].rolling(window=10).mean().plot(color='blue',label='WTCT4')
    ax.legend(loc='lower left')
    ax2 = ax.twinx()
    ax2 = df['Delac_' + str(i) + ' Diverter_Valve_Control_Valve_Feedback'].rolling(window=10).mean().plot(color='blue',linestyle='--',label='Diverter_Valve_Control_Valve_Feedback')
    ax2 = df['Delac_' + str(i) + ' Recirc_Fan_Motor_Requested_Speed'].rolling(window=10).mean().plot(color='red', linestyle='--', label='Recirc_Fan_Motor_Requested_Speed')
    ax.set_ylabel('Zone Temp')
    ax2.set_ylabel('Speed & diverter vlave')
    ax2.legend(loc='lower right')
    # plt.legend(loc='best')
    ax.set_ylim(300, 550)
    ax2.set_ylim(20, 100)
    ax.set(title=target + ' Trend')
    lst_stmt, df = save_fig(fig, target.replace(' ','_'))

    return lst_stmt, df
def RC_FAN_Vibration(df,i):
    target ='RC_Fan_Vibration_RC_fan_inlet_temp_'+str(i)
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    df = df.set_index('Timestamp')

    ax = df['Delac_' + str(i) + ' Recirculation_Fan_Inlet_Temp'].rolling(window=10).mean().plot(color='red',label='Recirculation_Fan_Inlet_Temp')
    ax.legend(loc='lower left')
    ax2 = ax.twinx()
    ax2 = df['Delac_' + str(i) + ' RC_FAN_Vibration'].rolling(window=10).mean().plot(color='blue',linestyle='--',label='RC_FAN_Vibration')
    ax.set_ylabel('Recirculation_Fan_Inlet_Temp')
    ax2.set_ylabel('RC_FAN_Vibration')
    ax2.legend(loc='lower right')
    # plt.legend(loc='best')
    ax.set_ylim(0, 400)
    ax2.set_ylim(0, 10)
    ax.set(title=target + ' Trend')
    lst_stmt, df = save_fig(fig, target)

    return lst_stmt, df
def RC_Fan_bearing_temp_pulley_fan(df):
    target ='RC_Fan_bearing_temp_pulley_fan'
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    df = df.set_index('Timestamp')
    df['Delac_1 RCFan_Pulley_Bearing_Temp'].plot(color='blue',label='Delac_1 RCFan_Pulley_Bearing_Temp')
    df['Delac_1 RCFan_Fan_Bearing_Temp'].plot(color='orange',label='Delac_1 RCFan_Fan_Bearing_Temp')
    df['Delac_2 RCFan_Pulley_Bearing_Temp'].plot(color='blue', linestyle='dashed',label='Delac_2 RCFan_Pulley_Bearing_Temp')
    df['Delac_2 RCFan_Fan_Bearing_Temp'].plot(color='orange', linestyle='dashed',label='Delac_2 RCFan_Fan_Bearing_Temp')
    ax.legend()
    ax.set(title=target + ' Trend')
    ax.set(ylim=(0, 80))
    ax.axhline(y=60, color='red', linestyle='dashed')
    lst_stmt, df = save_fig(fig, target)

    return lst_stmt, df
def support_roller_bearing_temp(df,i):
    target ='Support_Roller_Bearing_Temp_'+str(i)
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    df = df.set_index('Timestamp')
    df['Delac_{} Kiln_Bearing_1_Temp'.format(str(i))].plot(color='red',label='Delac_{} Kiln_Bearing_1_Temp'.format(str(i)))
    df['Delac_{} Kiln_Bearing_2_Temp'.format(str(i))].plot(color='blue',label='Delac_{} Kiln_Bearing_2_Temp'.format(str(i)))
    df['Delac_{} Kiln_Bearing_3_Temp'.format(str(i))].plot(color='green',label='Delac_{} Kiln_Bearing_3_Temp'.format(str(i)))
    df['Delac_{} Kiln_Bearing_4_Temp'.format(str(i))].plot(color='purple',label='Delac_{} Kiln_Bearing_4_Temp'.format(str(i)))
    df['Delac_{} Kiln_Bearing_5_Temp'.format(str(i))].plot(color='red', linestyle='dashed',
                                                           label='Delac_{} Kiln_Bearing_5_Temp'.format(str(i)))
    df['Delac_{} Kiln_Bearing_6_Temp'.format(str(i))].plot(color='blue', linestyle='dashed',
                                                           label='Delac_{} Kiln_Bearing_6_Temp'.format(str(i)))
    df['Delac_{} Kiln_Bearing_7_Temp'.format(str(i))].plot(color='green', linestyle='dashed',
                                                           label='Delac_{} Kiln_Bearing_7_Temp'.format(str(i)))
    df['Delac_{} Kiln_Bearing_8_Temp'.format(str(i))].plot(color='purple', linestyle='dashed',
                                                           label='Delac_{} Kiln_Bearing_8_Temp'.format(str(i)))

    ax.legend()
    ax.set(title=target + ' Trend')
    ax.set(ylim=(0, 80))
    ax.axhline(y=60, color='red', linestyle='dashed')
    lst_stmt, df = save_fig(fig, target)

    return lst_stmt, df
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
        if plot_info['plot'] =='line_alarm_multi':
            target_p = plot_info['target']
            Alarm = plot_info['Alarm']
        else:
            target_p =''
            Alarm = ''

        if lst_Out:
            avg_stmt += ', 관리범위 벗어난 기간: {}'.format(lst_Out)
        lst_avg.append(avg_stmt)
        path_plot, filename = create_visualization(target,df_v, x_col, y_col, lst_Out, plot,y_min,y_max,target_p,Alarm)
        files.append(filename)
    lst_stmt = list(zip(lst_avg, files))
    df_data = pd.DataFrame(list(zip(plot_type_map.keys(), lst_avg, files)),
                           columns=["Signal", "Stmt", "Plots"])
    df_data["Date"] = str(date.today())

    return lst_stmt,df_data


def get_plot_type_map(lst_Target,date_Out,avg_values):
    plot_type_map = {
        'Ti-B rod 투입 현황': {'x_col': 'Timestamp', 'y_col': 'DC_3 ROD_PV_TiBorSpeed',
                           'date_out': [],
                           'avg': avg_values['DC_3 ROD_PV_TiBorSpeed'], 'plot': 'line_raw_multi', 'y_min': 30,
                           'y_max': 60},
        'Debaler Bearing (DR NDR) 온도 Monitoring': {'x_col': 'Timestamp', 'y_col': 'SH BB_DS_BEARING_TEMP',
                                                   'date_out': [], 'avg': [], 'plot': 'line_alarm_multi',
                                                   'y_min': 0, 'y_max': 90, 'target': 70, 'Alarm': 85},
        '1 Shredder (DR NDR) 온도 Monitoring': {'x_col': 'Timestamp', 'y_col': 'SH RT_DS_BEARING_TEMP',
                                               'date_out': [], 'avg': [], 'plot': 'line_alarm_multi',
                                               'y_min': 0, 'y_max': 85, 'target': 70, 'Alarm': 80},
        '2 Shredder (DR NDR) 온도 Monitoring': {'x_col': 'Timestamp', 'y_col': 'SH HD_DS_BEARING_TEMP',
                                               'date_out': [], 'avg': [], 'plot': 'line_alarm_multi',
                                               'y_min': 0, 'y_max': 85, 'target': 70, 'Alarm': 80},
        'M22, M38 Conveyor 전류 Monitoring': {'x_col': 'Timestamp', 'y_col': 'M22_Amp_Out',
                                            'date_out': [], 'avg': [], 'plot': 'line_alarm_multi',
                                            'y_min': 0, 'y_max': 15, 'target': 9, 'Alarm': 10},
        'Fike system damper Monitoring_ Debaler': {'x_col': 'Timestamp', 'y_col': 'FIKE_BB DE_EIV1',
                                                   'date_out': [], 'avg': [], 'plot': 'line_alarm_multi',
                                                   'y_min': 0, 'y_max': 2, 'target': 0, 'Alarm': 2},
        'Fike system damper Monitoring_ #1 Shredder': {'x_col': 'Timestamp', 'y_col': 'FIKE_SH1 SH1_EIV1',
                                                       'date_out': [], 'avg': [], 'plot': 'line_alarm_multi',
                                                       'y_min': 0, 'y_max': 2, 'target': 0, 'Alarm': 2},
        'Fike system damper Monitoring_ #2 Shredder': {'x_col': 'Timestamp', 'y_col': 'FIKE_SH2 SH2_EIV1',
                                                       'date_out': [], 'avg': [], 'plot': 'line_alarm_multi',
                                                       'y_min': 0, 'y_max': 2, 'target': 0, 'Alarm': 2},
        'Debaler Shreder 1 2 대한 전류값 Monitoring': {'x_col': 'Timestamp', 'y_col': 'SH BB_AMPS',
                                                   'date_out': [], 'avg': [], 'plot': 'line_alarm_multi',
                                                   'y_min': 0, 'y_max': 204.5, 'target': 136.3, 'Alarm': 163.5},
        '주조 초기 용탕 온도': {'x_col': 'just_date', 'y_col': 'RT_1', 'date_out': date_Out['RT_1'], 'avg': avg_values['RT_1'],
                        'plot': 'box', 'y_min': 682, 'y_max': 3.5},
        '초기 냉각수 수온': {'x_col': 'just_date', 'y_col': 'CT_1', 'date_out': date_Out['CT_1'], 'avg': avg_values['CT_1'],
                      'plot': 'box', 'y_min': 30, 'y_max': 3.5},

        'decoater_1_3_zone_temp': {'x_col': 'just_date', 'y_col': 'Delac_1 WTCT3',
                                   'date_out': [], 'avg': avg_values['Delac_1 WTCT3'],
                                   'plot': 'box', 'y_min': 430, 'y_max': 3.5},
        'decoater_1_4_zone_temp': {'x_col': 'just_date', 'y_col': 'Delac_1 WTCT4',
                                   'date_out': [], 'avg': avg_values['Delac_1 WTCT4'],
                                   'plot': 'box', 'y_min': 540, 'y_max': 3.5},
        'decoater_2_3_zone_temp': {'x_col': 'just_date', 'y_col': 'Delac_2 WTCT3',
                                   'date_out': [], 'avg': avg_values['Delac_2 WTCT3'],
                                   'plot': 'box', 'y_min': 430, 'y_max': 3.5},
        'decoater_2_4_zone_temp': {'x_col': 'just_date', 'y_col': 'Delac_2 WTCT4',
                                   'date_out': [], 'avg': avg_values['Delac_2 WTCT4'],
                                   'plot': 'box', 'y_min': 540, 'y_max': 3.5},

        'Butt curl수준': {'x_col': 'just_date', 'y_col': 'BUTTCURL', 'date_out': date_Out['BUTTCURL'],
                        'avg': avg_values['BUTTCURL'], 'plot': 'box', 'y_min': 55, 'y_max': 55},
        'Alpur 염소 사용량': {'x_col': 'BATCHNO', 'y_col': 'Cl_Scale_Usage_Drop',
                         'date_out': date_Out['Cl_Scale_Usage_Drop'],
                         'avg': avg_values['Cl_Scale_Usage_Drop'], 'plot': 'bar', 'y_min': 0, 'y_max': 1},
        'Alpur 염소사용량(일자별)': {'x_col': 'just_date', 'y_col': 'Cl_Scale_Usage_Drop_Day',
                             'date_out': [],
                             'avg': avg_values['Cl_Scale_Usage_Drop_Day'], 'plot': 'bar', 'y_min': 0, 'y_max': 10},
        'Alpur Head Loss': {'x_col': 'BATCHNO', 'y_col': 'Alpur_head_loss', 'date_out': date_Out['Alpur_head_loss'],
                            'avg': avg_values['Alpur_head_loss'], 'plot': 'line', 'y_min': 20, 'y_max': 40},
        'DBF Head Loss': {'x_col': 'BATCHNO', 'y_col': 'DBF_head_loss', 'date_out': date_Out['DBF_head_loss'],
                          'avg': avg_values['DBF_head_loss'], 'plot': 'line', 'y_min': 40, 'y_max': 60},
        'Ca 제거효율': {'x_col': 'just_date', 'y_col': 'CA_REMOVE_RATE', 'date_out': [],
                    'avg': avg_values['CA_REMOVE_RATE'], 'plot': 'box', 'y_min': 40, 'y_max': 60},
        'Ca 제거효율(DROP)': {'x_col': 'BATCHNO', 'y_col': 'CA_REMOVE_RATE', 'date_out': [],
                          'avg': avg_values['CA_REMOVE_RATE'], 'plot': 'line', 'y_min': 40, 'y_max': 60},
        'Alpur 염소 저장소 Pressure': {'x_col': 'just_date', 'y_col': 'CT Cl2_Storage_Cl2_Pressure', 'date_out': [],
                                  'avg': avg_values['CT Cl2_Storage_Cl2_Pressure'], 'plot': 'line_raw', 'y_min': 2.5,
                                  'y_max': 3.5},
        'Alpur 염소 Main Panel Pressure': {'x_col': 'just_date', 'y_col': 'Alpur Cl_Main_Pressure',
                                         'date_out': [],
                                         'avg': avg_values['Alpur Cl_Main_Pressure'], 'plot': 'line_raw', 'y_min': 2.5,
                                         'y_max': 3.5},
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
                             'plot': 'line_raw', 'y_min': 50, 'y_max': 250},
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
        'Main differential pressure 7 Baghouse': {'x_col': 'Timestamp', 'y_col': 'Boiler PIT403-PIT402 _PRESS PIDC402',
                                                   'date_out': [],
                                                   'avg': 0,
                                                   'plot': 'line_raw', 'y_min': -75, 'y_max': -150},
        'Alpur leak': {'x_col': 'Timestamp', 'y_col': 'Alpur CHLORINE_MAIN.AI.Leak',
                       'date_out': [],
                       'avg': 0, 'plot': 'line_raw', 'y_min': -2, 'y_max': 2},
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
                         'avg': avg_values['Alpur TM.Heater1.Power_Mes'], 'plot': 'line_raw_multi', 'y_min': 10,
                         'y_max': 20},
        'DBF 예열': {'x_col': 'Timestamp', 'y_col': 'DBF_Pree TC_BoxTemp',
                   'date_out': [],
                   'avg': avg_values['DBF_Pree TC_BoxTemp'], 'plot': 'line_raw_TC_BoxTemp', 'y_min': 0, 'y_max': 700},
        'RFI 가동률': {'x_col': 'BATCHNO', 'y_col': 'Period', 'date_out': date_Out['Period'], 'avg': avg_values['Period'],
                    'plot': 'bar', 'y_min': 0, 'y_max': 20},
        '일자별_RFI_가동률': {'x_col': 'just_date', 'y_col': 'RFI_Day', 'date_out': [], 'avg': avg_values['RFI_Day'],
                        'plot': 'line', 'y_min': 0, 'y_max': 100},
        '월별_RFI_가동률': {'x_col': 'just_MONTH', 'y_col': 'RFI_Month', 'date_out': [], 'avg': avg_values['RFI_Month'],
                        'plot': 'line', 'y_min': 0, 'y_max': 100},
        'Split jet valve 압력 모니터링': {'x_col': 'Timestamp', 'y_col': 'DC_3 JET_PV_SplitJetFacePressure',
                                    'date_out': date_Out[''], 'avg': avg_values[''],
                                    'plot': 'line_raw', 'y_min': 2800, 'y_max': 3100},
        'Kiln & Cyclone Inlet temp_1': {'x_col': 'Timestamp', 'y_col': 'Delac_1 Cyclone_Inlet_Air_Temp_Setpoint',
                                    'date_out': date_Out[''], 'avg': avg_values[''],
                                    'plot': 'line_raw_multi', 'y_min': 0, 'y_max': 420},
        'Kiln & Cyclone Inlet temp_2': {'x_col': 'Timestamp', 'y_col': 'Delac_2 Cyclone_Inlet_Air_Temp_Setpoint',
                                      'date_out': date_Out[''], 'avg': avg_values[''],
                                      'plot': 'line_raw_multi', 'y_min': 0, 'y_max': 420},
        'Diverter Damper SP & PV_1': {'x_col': 'Timestamp', 'y_col': 'Delac_1 Diverter_Valve_Motor_Control_Signal',
                                        'date_out': date_Out[''], 'avg': avg_values[''],
                                        'plot': 'line_raw_multi', 'y_min': 0, 'y_max': 100},
        'Diverter Damper SP & PV_2': {'x_col': 'Timestamp', 'y_col': 'Delac_2 Diverter_Valve_Motor_Control_Signal',
                               'date_out': date_Out[''], 'avg': avg_values[''],
                               'plot': 'line_raw_multi', 'y_min': 0, 'y_max': 100},
        'Water Spray flow_1': {'x_col': 'Timestamp', 'y_col': 'Delac_1 WaterFlow_Afterburner_DayTot',
                               'date_out': date_Out[''], 'avg': avg_values[''],
                               'plot': 'line_raw_multi', 'y_min': 0, 'y_max': 200},
        'Water Spray flow_2': {'x_col': 'Timestamp', 'y_col': 'Delac_2 WaterFlow_Afterburner_DayTot',
                               'date_out': date_Out[''], 'avg': avg_values[''],
                               'plot': 'line_raw_multi', 'y_min': 0, 'y_max': 200},

        'Kiln_Driving_Motor': {'x_col': 'Timestamp', 'y_col': 'Delac_1 Kiln_Drive_Output_Current',
                          'date_out': date_Out[''], 'avg': avg_values[''],
                          'plot': 'line_raw_multi', 'y_min': 0, 'y_max': 70},

        'Support_Roller_Bearing_Temp_1': {'x_col': 'Timestamp', 'y_col': 'Delac_1 Kiln_Bearing_1_Temp',
                          'date_out': date_Out[''], 'avg': avg_values[''],
                          'plot': 'line_raw_multi', 'y_min': 10, 'y_max': 90, 'target':60},
        'Support_Roller_Bearing_Temp_2': {'x_col': 'Timestamp', 'y_col': 'Delac_2 Kiln_Bearing_1_Temp',
                                  'date_out': date_Out[''], 'avg': avg_values[''],
                                  'plot': 'line_raw_multi', 'y_min': 10, 'y_max': 90, 'target':60},
        'O2_lnlet_discharge_afterburner_1': {'x_col': 'Timestamp', 'y_col': 'Delac_1 Kiln_Inlet_O2',
                                  'date_out': date_Out[''], 'avg': avg_values[''],
                                  'plot': 'line_alarm_multi', 'y_min': 0, 'y_max': 25, 'target':7,'Alarm':0},
        'O2_lnlet_discharge_afterburner_2': {'x_col': 'Timestamp', 'y_col': 'Delac_2 Kiln_Inlet_O2',
                 'date_out': date_Out[''], 'avg': avg_values[''],
                 'plot': 'line_alarm_multi', 'y_min': 0, 'y_max': 25, 'target':7,'Alarm':0},
        'M24_40_Amp_Out': {'x_col': 'Timestamp', 'y_col': 'M24_Amp_Out',
                 'date_out': date_Out[''], 'avg': avg_values[''],
                 'plot': 'line_raw_multi', 'y_min': 0, 'y_max': 20},
        'M25_Amp_Out': {'x_col': 'Timestamp', 'y_col': 'M25_Amp_Out',
                           'date_out': date_Out[''], 'avg': avg_values[''],
                           'plot': 'line_raw', 'y_min': 0, 'y_max': 20},
        'Inlet_Airlock_Fault_1': {'x_col': 'Timestamp', 'y_col': 'Delac_1 Inlet_Airlock_Fault',
                           'date_out': date_Out[''], 'avg': avg_values[''],
                           'plot': 'line_raw_multi', 'y_min': -1, 'y_max': 2},
        'Inlet_Airlock_Fault_2': {'x_col': 'Timestamp', 'y_col': 'Delac_2 Inlet_Airlock_Fault',
                          'date_out': date_Out[''], 'avg': avg_values[''],
                          'plot': 'line_raw_multi', 'y_min': -1, 'y_max': 2},
        'Discharge_Airlock_Faulted_1': {'x_col': 'Timestamp', 'y_col': 'Delac_1 Discharge_Airlock_Faulted',
                                  'date_out': date_Out[''], 'avg': avg_values[''],
                                  'plot': 'line_raw_multi', 'y_min': -1, 'y_max': 2},
        'Discharge_Airlock_Faulted_2': {'x_col': 'Timestamp', 'y_col': 'Delac_2 Discharge_Airlock_Faulted',
                                  'date_out': date_Out[''], 'avg': avg_values[''],
                                  'plot': 'line_raw_multi', 'y_min': -1, 'y_max': 2},
        'Cyclone_Airlock_Fault_1': {'x_col': 'Timestamp', 'y_col': 'Delac_1 Cyclone_Airlock_Fault',
                                        'date_out': date_Out[''], 'avg': avg_values[''],
                                        'plot': 'line_raw_multi', 'y_min': -1, 'y_max': 2},
        'Cyclone_Airlock_Fault_2': {'x_col': 'Timestamp', 'y_col': 'Delac_2 Cyclone_Airlock_Fault',
                                        'date_out': date_Out[''], 'avg': avg_values[''],
                                        'plot': 'line_raw_multi', 'y_min': -1, 'y_max': 2},
        'Kiln_Debris_Airlock_Fault_1': {'x_col': 'Timestamp', 'y_col': 'Delac_1 Kiln_Debris_Airlock_Fault',
                                    'date_out': date_Out[''], 'avg': avg_values[''],
                                    'plot': 'line_raw_multi', 'y_min': -1, 'y_max': 2},
        'Kiln_Debris_Airlock_Fault_2': {'x_col': 'Timestamp', 'y_col': 'Delac_2 Kiln_Debris_Airlock_Fault',
                                    'date_out': date_Out[''], 'avg': avg_values[''],
                                    'plot': 'line_raw_multi', 'y_min': -1, 'y_max': 2},
        'Conveyor_Feedrate_PV_1': {'x_col': 'Timestamp', 'y_col': 'Delac_1 Conveyor_Feedrate_PV',
                                        'date_out': date_Out[''], 'avg': avg_values[''],
                                        'plot': 'line_raw_multi', 'y_min': 0, 'y_max': 30000},
        'Conveyor_Feedrate_PV_2': {'x_col': 'Timestamp', 'y_col': 'Delac_2 Conveyor_Feedrate_PV',
                                        'date_out': date_Out[''], 'avg': avg_values[''],
                                        'plot': 'line_raw_multi', 'y_min': 0, 'y_max': 30000},

        'GSS_CURRENT': {'x_col': 'Timestamp', 'y_col': 'SM1_GSS_CURRENT','date_out': date_Out[''], 'avg': avg_values[''],
        'plot': 'line_raw_multi', 'y_min': 0, 'y_max': 30},

        # 'Kiln_Driving_Motor_1': {'x_col': 'Timestamp', 'y_col': 'Delac_1 Kiln_Drive_Output_Current',
        #                            'date_out': date_Out[''], 'avg': avg_values[''],
        #                            'plot': 'line_raw', 'y_min': 0, 'y_max': 70},
        # 'Kiln_Driving_Motor_2': {'x_col': 'Timestamp', 'y_col': 'Delac_2 Kiln_Drive_Output_Current',
        #                            'date_out': date_Out[''], 'avg': avg_values[''],
        #                            'plot': 'line_raw', 'y_min': 0, 'y_max': 70},
    }
    # filtered dictionary
    filtered_map = {key: value for key, value in plot_type_map.items() if any(string in key for string in lst_Target)}
    return filtered_map

def create_visualization(target,df, x_col, y_col, lst_Out, plot,y_min,y_max,target_p,Alarm):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rc('font', size=20)  # controls default text sizes
    plt.rc('axes', titlesize=20)  # fontsize of the axes title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)  # fontsize of the tick labels
    plt.rc('legend', fontsize=20)  # legend fontsize
    plt.rc('figure', titlesize=20)  # fontsize of the figure title

    fig, ax = plt.subplots(figsize=(20, 10))
    if target in ["Floor cooling Air temp_1","Floor cooling Air temp_2","Floor cooling Air temp_3","Floor cooling Air temp_4"]:
        fig, ax = plt.subplots(figsize=(25, 15))
    if len(lst_Out) > 0:
        pal = {just_date: "r" if just_date in lst_Out else "b" for just_date in df.just_date.unique()}
    else:
        pal = sns.color_palette("mako_r", 6)
    #	FEEDRATE >= 15 TON
    if plot == 'box':
        if y_col[:12] == "Delac_1 WTCT":
            df=df.loc[df['Delac_1 Conveyor_Feedrate_PV']>=15000]
        elif y_col[:12] == "Delac_2 WTCT":
            df=df.loc[df['Delac_2 Conveyor_Feedrate_PV']>=15000]
        if target in ["Furnance pressure boxplot"]:
            box = sns.boxplot(x='variable', y='value', data=df,
                              order=['SM_1 aiFurnacePressure','SM_2 aiFurnacePressure','SM_3 aiFurnacePressure','SM_4 aiFurnacePressure','Holder PRESION_HORNO_mmCA'],
                              ax=ax, palette=pal)

            # ax.grid(True, linestyle='--', axis='y')
            # ax.set(ylim=(y_min, y_max))
            # box.axhline(y=target_p, color='red', linestyle='dashed')
        else:
            Q1 = df[y_col].quantile(0.25)
            Q3 = df[y_col].quantile(0.75)
            IQR = Q3 - Q1  # IQR is interquartile range.

            filter = (df[y_col] >= Q1 - 1.5 * IQR) & (df[y_col] <= Q3 + 1.5 * IQR)
            df=df.loc[filter]
                # box.axhline(y=target_p, color='red', linestyle='dashed')
            box = sns.boxplot(x=x_col, y=y_col, data=df, ax=ax, palette=pal)
        if y_col not in ["CA_REMOVE_RATE","SM_1 aiFurnacePressure"]:
            box.axhline(y=y_min, color='red', linestyle='dashed')

        box.set(title=target)
    elif plot == 'line':
        ax.set(title=y_col + ' Trend')
        # label points on the plot
        if y_col == 'DBF_head_loss':
            df = df.loc[df[y_col]!=-1]
        if x_col == 'BATCHNO':
            df = df.sort_values(x_col)
        for x, y , c in zip(df[x_col], df[y_col],df['check_' + y_col]):
            if c == 1:
                aColor='red'
            else:
                aColor = 'purple'
            plt.text(x=x, y=y, s='{:.1f}'.format(y), color=aColor)
        line = sns.lineplot(x=x_col, y=y_col, data=df, ax=ax, marker='o', palette=pal)
        if target not in('Ca 제거효율(DROP)', '일자별_RFI_가동률', '월별_RFI_가동률') :
            line.axhline(y=y_min, color='red', linestyle='dashed')
            line.axhline(y=y_max, color='red', linestyle='dashed')
        ax.tick_params(axis='x', rotation=90)
        ax.grid(True, linestyle='--', axis='y')
        line.set(title=target)

    elif plot == 'bar':
        ax.set(title=y_col + ' Trend')
        # label points on the plot
        bar = sns.barplot(x=x_col, y=y_col, data=df,ci=None)
        if y_col not in ["Cl_Scale_Usage_Drop_Day"]:
            bar.axhline(y=y_min, color='red', linestyle='dashed')
            bar.axhline(y=y_max, color='red', linestyle='dashed')
        if y_col in ["Cl_Scale_Usage_Drop_Day"]:
            ax.tick_params(axis='x', rotation=90)
        if target == 'RFI 가동률':
            lst_chk = []
            for x, y, c in zip(df['BATCHNO'], df['Period'], df['chk_time']):
                if c == True:
                    lst_chk.append(x)
            for label in ax.get_xticklabels():
                if label.get_text() in lst_chk:
                    label.set_color('red')

        bar.set(title=target)
        if y_col != ["Period"]:
            bar.axhline(y=y_max, color='red', linestyle='dashed')
        bar.bar_label(bar.containers[0], fmt='%.1f')
        ax.grid(True, linestyle='--', axis='y')

    elif plot == 'line_raw':
        if target == "Split jet valve 압력 모니터링":
            line = sns.lineplot(x=x_col, y=y_col, data=df)
            line.set(title=target)
            line.axhline(y=3100, color='red', linestyle='dashed')
            for x, c, v in zip(df[x_col], df['BATCHNO'].isnull(), df['BATCHNO']):
                if c == False:
                    plt.text(x=x, y=3150, rotation=90, s=v)
            ax.set(ylim=(2800, 3300))
        elif target in ['SM_1 BathTemperatureControl','SM_2 BathTemperatureControl','SM_3 BathTemperatureControl','SM_4 BathTemperatureControl'
                        ]:
            line = sns.lineplot(x=x_col, y=y_col, ax=ax, data=df)
            ax.set(ylim=(y_min, y_max))
            line.set(title=target)
        else:
            line = sns.lineplot(x=x_col, y=y_col, ax=ax, data=df)
            line.axhline(y=y_min, color='red', linestyle='dashed')
            line.axhline(y=y_max, color='red', linestyle='dashed')
            ax.grid(True, linestyle='--', axis='y')
            if target in ['소석회 투입량 체크','활성탄 투입량 체크','중탄산 투입량 체크']:
                ax.set(ylim=(df[y_col].min(), df[y_col].min() + 1000))
            elif target == 'Cooling tower cold pond level':
                ax.set(ylim=(20, 70))
            else:
                ax.set(ylim=(y_min-0.1, y_max+0.1))
            line.set(title=target)
    elif plot == 'line_raw_dot_multi':
        line = sns.lineplot(x=x_col, y='value', hue='variable', marker='o', markersize=10, data=pd.melt(df, [x_col]))
        for ycol in y_col:
            for x, y in zip(df[x_col], df[ycol]):
                plt.text(x=x, y=y, s='{:.1f}'.format(y), color='black',rotation=45,size=15)
        plt.xticks(rotation=45)
        ax.grid(True, linestyle='--', axis='y')
        line.set(title=target)
    elif plot == 'line_raw_multi':
        # line = sns.lineplot(x=x_col, y=y_col, ax=ax, data=df)
        line = sns.lineplot(x=x_col, y='value', hue='variable',data=pd.melt(df, [x_col]))
        ax.grid(True, linestyle='--', axis='y')

        if y_col == "Alpur CHLORINE.AI.Flow_Rotor_1":
            handles, labels = line.get_legend_handles_labels()
            handles = [h for i, h in enumerate(handles) if i not in (0,1)]
            labels = [l for i, l in enumerate(labels) if i not in (0,1)]
            line.legend(handles=handles, labels=labels)
            ax.set(ylim=(0, y_max + 10))
            line.axhline(y=y_min, color='red', linestyle='dashed')
            line.axhline(y=y_max, color='red', linestyle='dashed')
        elif target in ["노 바닥 온도","Floor cooling Air temp_1","Floor cooling Air temp_2","Floor cooling Air temp_3","Floor cooling Air temp_4","Water flow 온도"]:
            ax.set(ylim=(y_min, y_max))
            ax.tick_params(axis='x', rotation=45)
        elif target == 'PIT 301 & PIT 402 & PIT 403':
            ax.set(ylim=(y_min-10, y_max+10),yticks=range(y_min-10, y_max+10,50))
            line.axhline(y=y_min, color='red', linestyle='dashed')
            line.axhline(y=y_max, color='red', linestyle='dashed')
        elif target in ['Casting water supply pressure'
                        ,'Inlet_Airlock_Fault_1','Inlet_Airlock_Fault_2'
                        ,'Discharge_Airlock_Faulted_1','Discharge_Airlock_Faulted_2'
                        ,'Cyclone_Airlock_Fault_1','Cyclone_Airlock_Fault_2'
                        ,'Kiln_Debris_Airlock_Fault_1','Kiln_Debris_Airlock_Fault_2'
                        ,'No.5 Baghouse Main Fan Bearing temp'
                        ,'No.6 Baghouse Main Fan Bearing temp','SM1~4 Radar Level 모니터링','SM1~4 GSS RPM 모니터링'
                        ,'SM Bath or Roof mode 모니터링','SM_1 tcRoofTemperature','SM_2 tcRoofTemperature','SM_3 tcRoofTemperature','SM_4 tcRoofTemperature'
                        ,'Delac_1_Discharge_Airlock_Faulted','Delac_2_Discharge_Airlock_Faulted'
                        ]:
            ax.set(ylim=(y_min, y_max))
        elif target in ['RK 1&2 Gas Day Total']:
            ax.set(ylim=(y_min, y_max))
            for x, y in zip(df[x_col], df['Delac_1 Gas Day Tot 2_CST']):
                plt.text(x=x, y=y, s='{:.1f}'.format(y), color='black')
            for x, y in zip(df[x_col], df['Delac_2 Gas Day Tot 2_CST']):
                plt.text(x=x, y=y, s='{:.1f}'.format(y), color='black')
        else:
            ax.set(ylim=(y_min-10, y_max+10))
            line.axhline(y=y_min, color='red', linestyle='dashed')
            line.axhline(y=y_max, color='red', linestyle='dashed')


        line.set(title=target)
    elif plot == 'line_alarm_multi':
        if target in ['Sidewell melter furnance pressure trend','Holder furnance pressure trend']:
            line = sns.lineplot(x=x_col, y='value', hue='variable', marker='o',markersize=10, data=pd.melt(df, [x_col]))
            plt.xticks(rotation=45)

        else:
            line = sns.lineplot(x=x_col, y='value', hue='variable', data=pd.melt(df, [x_col]))
        # if target in ['MLC temp','DC_3 MLC_PV_AtlasLaserTemp']:
        #     line = sns.lineplot(x=x_col, y='value', hue='variable', data=pd.melt(df, [x_col]))
        # else:
        #     palette =['b', 'g','r']
        #     line = sns.lineplot(x=x_col, y='value', hue='variable', data=pd.melt(df, [x_col]), palette=palette)
        ax.grid(True, linestyle='--', axis='y')
        if target != 'M22, M38 Conveyor 전류 Monitoring':
            line.axhline(y=target_p, color='red', linestyle='dashed')
        line.axhline(y=Alarm, color='red',label='Alarm')
        if '온도' in target:
            line.set_ylabel("temp")
        elif '전류' in target:
            line.set_ylabel("전류값")
        # 'y축 0: 점선 및 "Semi Close" 문구 표시
        # 'y축 1: Open 문구 표시
        # 'y축 2: 점선 및 "Full Close" 문구 표시
        elif 'Fike' in target:
           line.set_ylabel("Open 여부")
           plt.yticks([0, 1, 2], ['Semi Close', 'Open', 'Full Close'])

        # line.text(0, Alarm + 10, f'Alarm', fontsize=13)
        # line.text(0, target_p + 10, f'Target', fontsize=13)
        plt.text(x=10, y=target_p, s='{:.1f}'.format(target_p))
        ax.set(ylim=(y_min, y_max))
        line.set(title=target)

    elif plot == 'line_raw_TC_BoxTemp':
        fig, ax = plt.subplots(1,2,figsize=(20, 10))
        df_month = df.loc[df.CAT == 'MONTH']
        df_month.drop('CAT', axis=1, inplace=True)
        df_today = df.loc[df.CAT == 'TODAY']
        df_today.drop('CAT', axis=1, inplace=True)
        # Month
        # plt.figure(figsize=(20, 10))
        plt.subplot(211)
        plt.title('Month', fontsize=15, pad=10)
        line_month= sns.lineplot(x=x_col, y='value', hue='variable', data=pd.melt(df_month, [x_col]))
        line_month.axhline(y=700, color='red', linestyle='dashed')
        line_month.set(title=target)
        plt.legend()
        # Today
        plt.subplot(212)
        # plt.title('Today', fontsize=15, pad=10)
        line_today = sns.lineplot(x=x_col, y='value', hue='variable', data=pd.melt(df_today, [x_col]))
        line_today.axhline(y=700, color='red', linestyle='dashed')
        plt.legend()
        # line_today.set(title=target)
    if target == 'Ca 제거효율(DROP)':
        y_col = 'CA_REMOVE_RATE_DROP'
    if target == 'RFI 가동률':
        y_col = 'New_RFI_Salt_Flow_PV'
    filename = str(date.today()) + "_" + target.replace(" ", "_").replace(".", "_") + ".png"
    # working directory
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    path_plot = str(dir) + folder + filename
    # save plot
    fig.savefig(path_plot, dpi=fig.dpi)

    return path_plot, filename

def remove_folder():
    command = ['databricks', 'fs', 'rm', '-r',
                                         'dbfs:/FileStore/User/changhyuck.lee@novelis.com/recycle_sig_mon/data']
    result = subprocess.run(command, capture_output=True, text=True)
    if result:
        print('data folder has removed')

def send_to_DBF(lst_stmt,data_file_path):
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    plot_path = str(dir) + folder
    # data_file_path = plot_path + data_file_path.split('/')[2]

    try:
        # command = ['databricks', 'fs', 'rm', '-r',
        #            'dbfs:/FileStore/User/changhyuck.lee@novelis.com/recycle_sig_mon/data']
        # result = subprocess.run(command, capture_output=True, text=True)
        # if result:
        #     print('data folder has removed')

        command = ['databricks', 'fs', 'cp', data_file_path,
                   'dbfs:/FileStore/User/changhyuck.lee@novelis.com/recycle_sig_mon/data']
        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True)
        logger.info("Data has sent successfully")
        print("Data has sent successfully")

        for f in lst_stmt:  # Add files to the message
            file_path = os.path.join(plot_path, f[1])
            # Build the CLI command
            command = ['databricks', 'fs', 'cp', file_path,
                       'dbfs:/FileStore/User/changhyuck.lee@novelis.com/recycle_sig_mon/data']
            # Execute the command
            result = subprocess.run(command, capture_output=True, text=True)
        logger.info("Plot has sent successfully")
        print("Plot has sent successfully")
    except Exception as err:
        print({"Error": str(err)})
        logger.info({"Error": str(err)})

def send_to_DBF2(name,df_data):
    try:
        str_today = str(date.today())
        # ./Data/ 폴더 경로 설정
        # folder_path = './Data/' + str_today
        #
        # # 일주일 전 날짜 계산
        # one_week_ago = date.today() - timedelta(days=7)
        # str_one_week_ago = str(one_week_ago)
        #
        # # 일주일 전 폴더 삭제
        # folder_to_delete = './Data/' + str_one_week_ago
        # if os.path.exists(folder_to_delete):
        #     os.rmdir(folder_to_delete)
        #
        # # 폴더 존재 여부 확인 및 생성
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        folder_path = './Data/'

        data_file_path = folder_path+'/{}.csv'.format(name)

        df_data.to_csv(data_file_path, index=False, encoding="utf-8-sig")
        # command = ['databricks', 'fs', 'mkdir',
        #            # 'dbfs:/FileStore/User/changhyuck.lee@novelis.com/recycle_wcm/data/' + str_today]
        #            'dbfs:/FileStore/User/changhyuck.lee@novelis.com/recycle_wcm/data/']
        # # Execute the command
        # result = subprocess.run(command, capture_output=True, text=True)

        command = ['databricks', 'fs', 'cp', '--overwrite', data_file_path, 'dbfs:/FileStore/User/changhyuck.lee@novelis.com/recycle_wcm/'+data_file_path]
        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True)
        logger.info("Data has sent successfully")
        print("Data has sent successfully")
    except Exception as err:
        print({"Error": str(err)})
        logger.info({"Error": str(err)})


# def send_to_SharePoint(lst_stmt):
#     dir = pathlib.Path(__file__).parent.absolute()
#     folder = r"/data/"
#     dir_path = str(dir) + folder
#     plot_path = str(dir) + r"/data/"
#     str_today = str(date.today())
#     data_file_path = dir_path+'/data_{}.csv'.format(str_today)
#     folder_path = str_today
#     # upload Data file
#     SharePoint().upload_file(data_file_path, 'df_M_{}.csv'.format(str_today), folder_path)
#     SharePoint().upload_file(data_file_path, 'data_{}.csv'.format(str_today), folder_path)
#     logger.info("Data has sent to SharePoint successfully")
#     for f in lst_stmt:  # Add files to the message
#         plots_file_name = f[1]
#         plots_file_path = os.path.join(plot_path, plots_file_name)
#     # upload Plot file
#         SharePoint().upload_file(plots_file_path, plots_file_name, folder_path)
#
#     logger.info("Plot has sent to SharePoint successfully")
#
#     # delete file
#     # SharePoint().delete_file(file_name, folder_name)

# send email using subject and body
def send_email_multi(recipients, lst_Target, lst_stmt):

    Targets = ' '
    for Target in lst_Target:
        Targets += ' ' + Target

    subject = Targets + ' Trend'
    body = "Dear All, "  # Graph File generation
    body += "<br>" + Targets + " trend 알려 드립니다."
    # body += "<br>" + "Data 추출기간 : [{} ~ {}] ".format(start_date, end_date)
    # smtp_server = "smtp.gmail.com"
    # smtp_port = 465
    server = smtplib.SMTP('smtp.agdc.novelis.com', 25)
    # server = smtplib.SMTP_SSL(smtp_server, smtp_port)
    server.ehlo()
    # server.login("victorlee12@gmail.com", "from_password")
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = "NoReply@novelis.com"
    msg['To'] = ",".join(recipients)
    # Attach HTML body
    #   body += lst_stmt #"<br>"+Target + " 평균 {}".format(avg_Col)
    dir_path = pathlib.Path(__file__).parent.absolute() +r"/plots/"

    imsg_src = ""
    for f in lst_stmt:  # Add files to the message
        file_path = os.path.join(dir_path, f[1])
        attachment = MIMEApplication(open(file_path, "rb").read(), _subtype="png")
        attachment.add_header('Content-Disposition', 'attachment', filename=f[1])
        msg.attach(attachment)
        imsg_src += f[0] + '''<br><img src=''' + f[1] + '''></br>'''

    msg.attach(MIMEText(
        '''
        <html>
            <body>
                <p>''' + body + '''</p>
              ''' + imsg_src + '''
          </body>
      </html>'
      ''',
        'html', 'utf-8'))

    server.sendmail("NoReply@novelis.com", recipients, msg.as_string())
    server.close()
from PIL import Image

def shrink_image(input_path, output_path, max_size):
    # Open the image file
    image = Image.open(input_path)

    # Calculate the new width and height while maintaining the aspect ratio
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int((height / width) * max_size)
    else:
        new_width = int((width / height) * max_size)
        new_height = max_size

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Save the resized image
    resized_image.save(output_path)

def retention_file():
    from datetime import date
    from datetime import timedelta
    import os
    import shutil
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    mypath = str(dir) + folder
    today = date.today()
    yesterday = today - timedelta(days=1)
    yesterday = yesterday.strftime("%Y-%m-%d")

    del_day = today - timedelta(days=2)
    del_day = del_day.strftime("%Y-%m-%d")
    del_day = mypath + '/' + del_day

    if not os.path.exists(mypath + yesterday):
        os.mkdir(mypath + yesterday)
        # move yesterday file to folder
        for f in os.listdir(mypath):
            if os.path.isfile(os.path.join(mypath, f)):
                if f.split('_')[0] == yesterday:
                    shutil.move(mypath + f, mypath + '/' + yesterday + '/' + f)

    try:
        shutil.rmtree(del_day)
    except Exception as e:
        print(e)

def DoHee(df_raw):
    lst_Target_DoHee = [
        'Alpur Head Loss',
        'DBF Head Loss',
        '주조 초기 용탕 온도',
        '초기 냉각수 수온',
        'Butt curl수준',
    ]
    # lst_recipients_DoHee = ['ChangHyuck Lee <ChangHyuck.Lee@novelis.adityabirla.com>']
    lst_recipients_DoHee = ['ChangHyuck Lee <ChangHyuck.Lee@novelis.adityabirla.com>',
                            "HongChul Kim <HongChul.Kim@novelis.adityabirla.com>", "yun.lee@novelis.adityabirla.com",
                            "DoHee.Kim <DoHee.Kim1@novelis.adityabirla.com>",
                            "WonHun Song <WonHun.Song@novelis.adityabirla.com>",
                            "JooSeop Jung <JooSeop.Jung@novelis.adityabirla.com>",
                            "DongRyeol Lee <DongRyeol.Lee@novelis.adityabirla.com>",
                            "WookSeob Hwang <WookSeob.Hwang@novelis.adityabirla.com>",
                            "HanSub Kim <hansub.kim@novelis.adityabirla.com>",
                            "ByoungGyu Jang <byounggyu.jang@novelis.adityabirla.com>",
                            "ByungYong Kang <KangByungyon@novelis.adityabirla.com>",
                            "HyeongSik Moon <HyeongSik.Moon@novelis.adityabirla.com>",
                            "JinHyun Bang <jinhyun.bang@novelis.adityabirla.com>",
                            "SungHo Jung <SungHo.Jung@novelis.adityabirla.com>",
                            "DongHoon Kim <donghoon.kim@novelis.adityabirla.com>",
                            "HaeIn Yun <HaeIn.Yun@novelis.adityabirla.com>",
                            "HyukDo Kwon <HyukDo.Kwon@novelis.adityabirla.com>",
                            "Seunghyun Um <SeungHyun.Um@novelis.adityabirla.com>",
                            "YoungSik Lee <YoungSik.Lee@novelis.adityabirla.com>",
                            "JunYong.Lee@novelis.adityabirla.com",
                            "JoongHyo Lee <JoongHyo.Lee@novelis.adityabirla.com>",
                            "Changbeom Jho <changbeom.jho@novelis.adityabirla.com>",
                            "MiHee Kim <MiHee.Kim@novelis.adityabirla.com>",
                            "JiYu Kim <JiYu.Kim@novelis.adityabirla.com>",
                            "WonHo Bae <WonHo.Bae@novelis.adityabirla.com>",
                            "JeongKeun Jo <JeongKeun.Jo@novelis.adityabirla.com>",
                            "KiBum Kim <KiBum.Kim@novelis.adityabirla.com>"
                            ]
    df_raw_DoHee = df_raw.loc[df_raw.Signal.isin(lst_Target_DoHee)]
    lst_stmt_DoHee = list(zip(df_raw_DoHee['Stmt'].to_list(), df_raw_DoHee['Plots'].to_list()))

    # send_email_path(lst_recipients_DoHee, lst_Target_DoHee, raw_path, lst_stmt_DoHee)
    send_email_multi(lst_recipients_DoHee, lst_Target_DoHee, lst_stmt_DoHee)
def ChangBum(df_raw):
    lst_Target_ChangBum = [
        'Ti-B rod 투입 현황',
        'DBF 예열',
        'Cooling Tower',
        'Casting pit water level', 'Cooling tower cold pond level',
        'Casting water supply pressure', 'Casting water supply flow',
        'Split jet valve 압력 모니터링']
    # lst_recipients_ChangBum = ['ChangHyuck Lee <ChangHyuck.Lee@novelis.adityabirla.com>']
    lst_recipients_ChangBum = ['ChangHyuck Lee <ChangHyuck.Lee@novelis.adityabirla.com>',
                               "yun.lee@novelis.adityabirla.com", "changbeom.jho@novelis.adityabirla.com",
                               "DoHee.Kim1@novelis.adityabirla.com", "donghoon.kim@novelis.adityabirla.com",
                               "HaeIn.Yun@novelis.adityabirla.com", "HongChul.Kim@novelis.adityabirla.com",
                               "HyukDo.Kwon@novelis.adityabirla.com", "JeongKeun.Jo@novelis.adityabirla.com",
                               "JiYu.Kim@novelis.adityabirla.com", "JoongHyo.Lee@novelis.adityabirla.com",
                               "JunYong.Lee@novelis.adityabirla.com", "MiHee.Kim@novelis.adityabirla.com",
                               "SeungHyun.Um@novelis.adityabirla.com", "WonHo.Bae@novelis.adityabirla.com",
                               "YoungSik.Lee@novelis.adityabirla.com", "byounggyu.jang@novelis.adityabirla.com",
                               "KangByungyon@novelis.adityabirla.com", "hansub.kim@novelis.adityabirla.com",
                               "KiBum.Kim@novelis.adityabirla.com", "WonHun.Song@novelis.adityabirla.com",
                               "WookSeob.Hwang@novelis.adityabirla.com", "SungHo.Jung@novelis.adityabirla.com",
                               "jinhyun.bang@novelis.adityabirla.com", "JooSeop.Jung@novelis.adityabirla.com",
                               "KimKyutae@novelis.adityabirla.com", 'DongRyeol.Lee@novelis.adityabirla.com']
    df_raw_ChangBum = df_raw.loc[df_raw.Signal.isin(lst_Target_ChangBum)]
    lst_stmt_ChangBum = list(zip(df_raw_ChangBum['Stmt'].to_list(), df_raw_ChangBum['Plots'].to_list()))
    send_email_multi(lst_recipients_ChangBum, lst_Target_ChangBum, lst_stmt_ChangBum)

def Alpur(df_raw):
    lst_Target_Alpur = [
        'Alpur 염소 사용량 & Ca 제거 효율(Drop)',
        'Alpur 염소 사용량',
        'Alpur 염소사용량(일자별)',
        'Alpur leak',
        'Ca 제거효율',
        'Ca 제거효율(DROP)',
        'Alpur 염소 Flow',
        'Alpur 염소 저장소 Pressure',
        'Alpur 염소 Main Panel Pressure',
        'Heater Power',
        'Alpur leak']
    lst_recipients_Alpur = ['ChangHyuck Lee <ChangHyuck.Lee@novelis.adityabirla.com>',
                            "yun.lee@novelis.adityabirla.com", "changbeom.jho@novelis.adityabirla.com",
                            "DoHee.Kim1@novelis.adityabirla.com", "donghoon.kim@novelis.adityabirla.com",
                            "HaeIn.Yun@novelis.adityabirla.com", "HongChul.Kim@novelis.adityabirla.com",
                            "HyukDo.Kwon@novelis.adityabirla.com", "JeongKeun.Jo@novelis.adityabirla.com",
                            "JiYu.Kim@novelis.adityabirla.com", "JoongHyo.Lee@novelis.adityabirla.com",
                            "JunYong.Lee@novelis.adityabirla.com", "MiHee.Kim@novelis.adityabirla.com",
                            "SeungHyun.Um@novelis.adityabirla.com", "WonHo.Bae@novelis.adityabirla.com",
                            "YoungSik.Lee@novelis.adityabirla.com", "byounggyu.jang@novelis.adityabirla.com",
                            "KangByungyon@novelis.adityabirla.com", "hansub.kim@novelis.adityabirla.com",
                            "KiBum.Kim@novelis.adityabirla.com", "WonHun.Song@novelis.adityabirla.com",
                            "SungHo.Jung@novelis.adityabirla.com", "jinhyun.bang@novelis.adityabirla.com",
                            "JooSeop.Jung@novelis.adityabirla.com", "KimKyutae@novelis.adityabirla.com",
                            'DongRyeol.Lee@novelis.adityabirla.com']

    df_raw_Alpur = df_raw.loc[df_raw.Signal.isin(lst_Target_Alpur)]
    lst_stmt_Alpur = list(zip(df_raw_Alpur['Stmt'].to_list(), df_raw_Alpur['Plots'].to_list()))
    lst_stmt_Alpur_sorted = sorted(lst_stmt_Alpur, key=lambda x: lst_Target_Alpur.index(x[0].split(' 평균')[0]))
    send_email_multi(lst_recipients_Alpur, lst_Target_Alpur, lst_stmt_Alpur_sorted)

def RFI(df_raw):
    lst_Target_RFI = [
        'RFI 가동률',
        '일자별_RFI_가동률'
    ]
    df_raw_RFI = df_raw.loc[df_raw.Signal.isin(lst_Target_RFI)]
    lst_stmt_RFI = list(zip(df_raw_RFI['Stmt'].to_list(), df_raw_RFI['Plots'].to_list()))
    lst_recipients_Alpur = ['ChangHyuck Lee <ChangHyuck.Lee@novelis.adityabirla.com>',
                            "yun.lee@novelis.adityabirla.com", "changbeom.jho@novelis.adityabirla.com",
                            "DoHee.Kim1@novelis.adityabirla.com", "donghoon.kim@novelis.adityabirla.com",
                            "HaeIn.Yun@novelis.adityabirla.com", "HongChul.Kim@novelis.adityabirla.com",
                            "HyukDo.Kwon@novelis.adityabirla.com", "JeongKeun.Jo@novelis.adityabirla.com",
                            "JiYu.Kim@novelis.adityabirla.com", "JoongHyo.Lee@novelis.adityabirla.com",
                            "JunYong.Lee@novelis.adityabirla.com", "MiHee.Kim@novelis.adityabirla.com",
                            "SeungHyun.Um@novelis.adityabirla.com", "WonHo.Bae@novelis.adityabirla.com",
                            "YoungSik.Lee@novelis.adityabirla.com", "byounggyu.jang@novelis.adityabirla.com",
                            "KangByungyon@novelis.adityabirla.com", "hansub.kim@novelis.adityabirla.com",
                            "KiBum.Kim@novelis.adityabirla.com", "WonHun.Song@novelis.adityabirla.com",
                            "SungHo.Jung@novelis.adityabirla.com", "jinhyun.bang@novelis.adityabirla.com",
                            "JooSeop.Jung@novelis.adityabirla.com", "KimKyutae@novelis.adityabirla.com",
                            'DongRyeol.Lee@novelis.adityabirla.com']
    send_email_multi(lst_recipients_Alpur, lst_Target_RFI, lst_stmt_RFI)

def decoater(df_raw):
    lst_Target_decoater = ['decoater 1 #3zone temp', 'decoater 2 #3zone temp', 'decoater 1 #4zone temp',
                           'decoater 2 #4zone temp', '노압 압력 모니터링', 'PIT 301 & PIT 402 & PIT 403']
    df_raw_test = df_raw.loc[df_raw.Signal.isin(lst_Target_decoater)]
    lst_stmt_decoater = list(zip(df_raw_test['Stmt'].to_list(), df_raw_test['Plots'].to_list()))
    # 순서를 #1 decoater 3zone temp  #2 decoater  3zone temp  #1 decoater 4 zone temp  #2 decoater 4zone temp 이와 같은 순서로 회람
    # 정렬 순서는 `#1 decoater`, `#2 decoater`, `3zone`, `4zone`, `temp` 순서
    lst_stmt_decoater.sort(key=lambda x: (x[0].split()[1], x[0].split()[0], x[0].split()[2][:1]))

    recipients_decoater = ['ChangHyuck Lee <ChangHyuck.Lee@novelis.adityabirla.com>', "yun.lee@novelis.adityabirla.com",
                           'Changbeom Jho <changbeom.jho@novelis.adityabirla.com>',
                           ' DoHee.Kim <DoHee.Kim1@novelis.adityabirla.com>',
                           ' DongHoon Kim <donghoon.kim@novelis.adityabirla.com>',
                           ' HaeIn Yun <HaeIn.Yun@novelis.adityabirla.com>',
                           ' HongChul Kim <HongChul.Kim@novelis.adityabirla.com>',
                           ' HyukDo Kwon <HyukDo.Kwon@novelis.adityabirla.com>',
                           ' JeongKeun Jo <JeongKeun.Jo@novelis.adityabirla.com>',
                           ' JiYu Kim <JiYu.Kim@novelis.adityabirla.com>',
                           ' JoongHyo Lee <JoongHyo.Lee@novelis.adityabirla.com>',
                           ' JunYong Lee <JunYong.Lee@novelis.adityabirla.com>',
                           ' MiHee Kim <MiHee.Kim@novelis.adityabirla.com>',
                           ' Seunghyun Um <SeungHyun.Um@novelis.adityabirla.com>',
                           ' WonHo Bae <WonHo.Bae@novelis.adityabirla.com>',
                           ' YoungSik Lee <YoungSik.Lee@novelis.adityabirla.com>',
                           ' ByoungGyu Jang <byounggyu.jang@novelis.adityabirla.com>',
                           ' ByungYong Kang <KangByungyon@novelis.adityabirla.com>',
                           ' DongRyeol Lee <DongRyeol.Lee@novelis.adityabirla.com>',
                           ' HanSub Kim <hansub.kim@novelis.adityabirla.com>',
                           ' JaeHyeon Kwon <Jaehyeon.Kwon@novelis.adityabirla.com>',
                           ' JooSeop Jung <JooSeop.Jung@novelis.adityabirla.com>',
                           ' KiBum Kim <KiBum.Kim@novelis.adityabirla.com>',
                           ' KyuTae Kim <KimKyutae@novelis.adityabirla.com>',
                           ' SeokJu Seo <seokju.seo@novelis.adityabirla.com>',
                           ' SeongIn Jung <seongin.jung@novelis.adityabirla.com>',
                           ' SooDong An <SooDong.An@novelis.adityabirla.com>',
                           ' WonHun Song <WonHun.Song@novelis.adityabirla.com>',
                           ' WooJeong Lee <WooJeong.Lee@novelis.adityabirla.com>',
                           ' YangUk Shon <YangUk.Shon@novelis.adityabirla.com>'
                           ]
    lst_stmt_decoater.sort(key=lambda x: (x[0].split()[1], x[0].split()[0], x[0].split()[2][:1]))
    send_email_multi(recipients_decoater, lst_Target_decoater, lst_stmt_decoater)

def target_1(df_raw):
    lst_Target_1 = ['Main differential pressure', '활성탄 투입량 체크', 'ph값/가성소다/소석회/중탄산',
                    ]

    lst_recipients_1 = ['ChangHyuck Lee <ChangHyuck.Lee@novelis.adityabirla.com>', "yun.lee@novelis.adityabirla.com",
                        'JeongKeun.Jo@novelis.adityabirla.com', 'HongChul.Kim@novelis.adityabirla.com',
                        'JongIk.Kim@novelis.adityabirla.com',
                        'byounggyu.jang@novelis.adityabirla.com',
                        'KangByungyon@novelis.adityabirla.com',
                        'changbeom.jho@novelis.adityabirla.com',
                        'ChanSu.Lim@novelis.adityabirla.com',
                        'DoHee.Kim1@novelis.adityabirla.com',
                        'donghoon.kim@novelis.adityabirla.com',
                        'DongRyeol.Lee@novelis.adityabirla.com',
                        'gumin.kwon@novelis.adityabirla.com',
                        'gyeonghwan.kim@novelis.adityabirla.com',
                        'HaeIn.Yun@novelis.adityabirla.com',
                        'hansub.kim@novelis.adityabirla.com	 ',
                        'HyukDo.Kwon@novelis.adityabirla.com',
                        'jaeheum.lee@novelis.adityabirla.com',
                        'Jaehyeon.Kwon@novelis.adityabirla.com',
                        'JeongKeun.Jo@novelis.adityabirla.com',
                        'jinhyun.bang@novelis.adityabirla.com',
                        'JiYu.Kim@novelis.adityabirla.com',
                        'jongu.kim@novelis.adityabirla.com',
                        'JoongHyo.Lee@novelis.adityabirla.com',
                        'JooSeop.Jung@novelis.adityabirla.com',
                        'JungSub.Kim@novelis.adityabirla.com',
                        'JunYong.Lee@novelis.adityabirla.com',
                        'kihong.kwon1@novelis.adityabirla.com',
                        'KimKyutae@novelis.adityabirla.com',
                        'MiHee.Kim@novelis.adityabirla.com',
                        'SeokJu.Cha@novelis.adityabirla.com',
                        'seokju.seo@novelis.adityabirla.com',
                        'SeongHo.Son@novelis.adityabirla.com',
                        'seonghyen.jang@novelis.adityabirla.com',
                        'seongin.jung@novelis.adityabirla.com',
                        'SeongJik.Bae@novelis.adityabirla.com',
                        'SeungHyun.Um@novelis.adityabirla.com',
                        'SooDong.An@novelis.adityabirla.com',
                        'sookyoung.kang@novelis.adityabirla.com',
                        'sukhoon.keum@novelis.adityabirla.com',
                        'sungryong.kang@novelis.adityabirla.com',
                        'SuSang.Han@novelis.adityabirla.com',
                        'WonHo.Bae@novelis.adityabirla.com',
                        'WonHun.Song@novelis.adityabirla.com',
                        'WookSeob.Hwang@novelis.adityabirla.com',
                        'YangUk.Shon@novelis.adityabirla.com',
                        'YoungSik.Lee@novelis.adityabirla.com',
                        ]

    df_raw_1 = df_raw.loc[df_raw.Signal.isin(lst_Target_1)]
    lst_stmt_1 = list(zip(df_raw_1['Stmt'].to_list(), df_raw_1['Plots'].to_list()))
    lst_stmt_1.sort()
    send_email_multi(lst_recipients_1,lst_Target_1,lst_stmt_1)

def target_2(df_raw):
    raw_path = "/dbfs/FileStore/User/changhyuck.lee@novelis.com/recycle_sig_mon/data/"
    lst_Target_2 = ['Debaler Bearing (DR/NDR) 온도 Monitoring', '#1 Shredder (DR/NDR) 온도 Monitoring',
                    '#2 Shredder (DR/NDR) 온도 Monitoring',
                    'M22, M38 Conveyor 전류 Monitoring', 'Fike system damper Monitoring_ Debaler',
                    'Fike system damper Monitoring_ #1 Shredder', 'Fike system damper Monitoring_ #2 Shredder',
                    'Debaler, Shreder1, 2 대한 전류값 Monitoring']

    lst_recipients_2 = ['ChangHyuck Lee <ChangHyuck.Lee@novelis.adityabirla.com>', "yun.lee@novelis.adityabirla.com",
                        'Changbeom.jho@novelis.adityabirla.com ', 'Jongu.kim@novelis.adityabirla.com',
                        'DoHee.Kim1@novelis.adityabirla.com',
                        'donghoon.kim@novelis.adityabirla.com',
                        'HaeIn.Yun@novelis.adityabirla.com',
                        'HongChul.Kim@novelis.adityabirla.com ',
                        'HyukDo.Kwon@novelis.adityabirla.com',
                        'JeongKeun.Jo@novelis.adityabirla.com',
                        'JiYu.Kim@novelis.adityabirla.com',
                        'JoongHyo.Lee@novelis.adityabirla.com',
                        'JunYong.Lee@novelis.adityabirla.com',
                        'MiHee.Kim@novelis.adityabirla.com',
                        'SeungHyun.Um@novelis.adityabirla.com ',
                        'WonHo.Bae@novelis.adityabirla.com ',
                        'YoungSik.Lee@novelis.adityabirla.com',
                        'Jongu.kim@novelis.adityabirla.com',
                        'JooSeop.Jung@novelis.adityabirla.com',
                        'WonHun.Song@novelis.adityabirla.com',
                        ]
    df_raw_2 = df_raw.loc[df_raw.Signal.isin(lst_Target_2)]
    lst_stmt_2 = list(zip(df_raw_2['Stmt'].to_list(), df_raw_2['Plots'].to_list()))
    send_email_multi(lst_recipients_2, lst_Target_2, raw_path, lst_stmt_2)

def send_email_seq(df_raw):
    try:
        DoHee(df_raw)

        # ChangBum(df_raw)
        # Alpur(df_raw)
        # RFI(df_raw)
        # decoater(df_raw)
        # target_1(df_raw)
        # target_2(df_raw)
    except Exception as err:
        print({"Error": str(err)})
        logger.info({"Error": str(err)})
        pass

def send_email_Error(error):

    subject = 'Error happend at '+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+ "<br>" + error
    body = "<br>" + error
    server = smtplib.SMTP('relay.novelis.com', 25)
    server.ehlo()
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = "NoReply@novelis.com"
    msg['To'] = "changhyuck.lee@novelis.com"
    server.sendmail("NoReply@novelis.com", "changhyuck.lee@novelis.com", msg.as_string())
    server.close()


