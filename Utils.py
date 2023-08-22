from datetime import datetime, date
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
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))

    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels

    fig, ax = plt.subplots(1,2,figsize=(30, 20))

    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    plt.subplot(211)
    plt.title('Decoater 1')
    plt.plot(df['Timestamp'],df['Delac_1 Bag_Pressure_Transmitter PIT111'], label='Bag_Pressure_Transmitter PIT111', color='red')
    plt.plot(df['Timestamp'],df['Delac_1 System_Pressure_Valve_Motor_Control_Sig'], label='System_Pressure_Valve_Motor_Control_Sig', color='blue')
    plt.plot(df['Timestamp'],df['Delac_1 System_Pressure_Control_Valve_Feedback'], label='System_Pressure_Control_Valve_Feedback')
    plt.subplot(211).xaxis.set_major_locator(md.MinuteLocator(byminute=[0, 60]))
    plt.subplot(211).yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    plt.subplot(211).set_ylim([-20, 100])
    plt.tick_params(axis='x', rotation=45, labelsize=15)
    plt.legend()
    plt.subplot(212)
    plt.title('Decoater 2')
    plt.plot(df['Timestamp'],df['Delac_2 Bag_Pressure_Transmitter PIT121'], label='Bag_Pressure_Transmitter PIT121', color='red')
    plt.plot(df['Timestamp'],df['Delac_2 System_Pressure_Valve_Motor_Control_Sig'],
             label='System_Pressure_Valve_Motor_Control_Sig', color='blue')
    plt.plot(df['Timestamp'],df['Delac_2 System_Pressure_Control_Valve_Feedback'],
             label='System_Pressure_Control_Valve_Feedback')
    plt.subplot(212).xaxis.set_major_locator(md.MinuteLocator(byminute=[0, 60]))
    plt.subplot(212).yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    plt.subplot(212).set_ylim([-20, 100])
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

def create_visualization(target,df, x_col, y_col, lst_Out, plot,y_min,y_max,target_p,Alarm):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rc('font', size=15)  # controls default text sizes
    plt.rc('axes', titlesize=20)  # fontsize of the axes title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=15)  # fontsize of the tick labels
    plt.rc('legend', fontsize=15)  # legend fontsize
    plt.rc('figure', titlesize=20)  # fontsize of the figure title

    fig, ax = plt.subplots(figsize=(20, 10))
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

        Q1 = df[y_col].quantile(0.25)
        Q3 = df[y_col].quantile(0.75)
        IQR = Q3 - Q1  # IQR is interquartile range.

        filter = (df[y_col] >= Q1 - 1.5 * IQR) & (df[y_col] <= Q3 + 1.5 * IQR)
        df=df.loc[filter]
        box = sns.boxplot(x=x_col, y=y_col, data=df, ax=ax, palette=pal)
        if y_col not in ["CA_REMOVE_RATE"]:
            box.axhline(y=y_min, color='red', linestyle='dashed')
        box.set(title=target)
    elif plot == 'line':
        ax.set(title=y_col + ' Trend')
        # label points on the plot
        if y_col == 'DBF_head_loss':
            df = df.loc[df[y_col]!=-1]
        for x, y , c in zip(df[x_col], df[y_col],df['check_' + y_col]):
            if c == 1:
                aColor='red'
            else:
                aColor = 'purple'
            plt.text(x=x, y=y, s='{:.1f}'.format(y), color=aColor)
        line = sns.lineplot(x=x_col, y=y_col, data=df, ax=ax, marker='o', palette=pal)
        if target not in('Ca 제거효율(DROP)', '일자별_RFI_가동률') :
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
    elif plot == 'line_raw_multi':
        # line = sns.lineplot(x=x_col, y=y_col, ax=ax, data=df)
        line = sns.lineplot(x=x_col, y='value', hue='variable',data=pd.melt(df, [x_col]))
        ax.grid(True, linestyle='--', axis='y')
        line.axhline(y=y_min, color='red', linestyle='dashed')
        line.axhline(y=y_max, color='red', linestyle='dashed')
        if y_col == "Alpur CHLORINE.AI.Flow_Rotor_1":
            handles, labels = line.get_legend_handles_labels()
            handles = [h for i, h in enumerate(handles) if i not in (0,1)]
            labels = [l for i, l in enumerate(labels) if i not in (0,1)]
            line.legend(handles=handles, labels=labels)
            ax.set(ylim=(0, y_max + 10))
        elif target == 'PIT 301 & PIT 402 & PIT 403':
            ax.set(ylim=(y_min-10, y_max+10),yticks=range(y_min-10, y_max+10,50))
        elif target == 'Casting water supply pressure':
            ax.set(ylim=(y_min, y_max))
        else:
            ax.set(ylim=(y_min-10, y_max+10))

        line.set(title=target)
    elif plot == 'line_alarm_multi':
        line = sns.lineplot(x=x_col, y='value', hue='variable', data=pd.melt(df, [x_col]), palette=['b', 'g'])
        ax.grid(True, linestyle='--', axis='y')
        if target != 'M22, M38 Conveyor 전류 Monitoring':
            line.axhline(y=target_p, color='red', linestyle='dashed')
        line.axhline(y=Alarm, color='red')
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
    filename = str(date.today()) + "_" + y_col.replace(" ", "_") + ".png"
    # working directory
    dir = pathlib.Path(__file__).parent.absolute()
    folder = r"/data/"
    path_plot = str(dir) + folder + filename
    # save plot
    fig.savefig(path_plot, dpi=fig.dpi)

    # max_image_size = 500  # Maximum width or height
    # shrink_image(path_plot, path_plot, max_image_size)

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

        for f in lst_stmt:  # Add files to the message
            file_path = os.path.join(plot_path, f[1])
            # Build the CLI command
            command = ['databricks', 'fs', 'cp', file_path,
                       'dbfs:/FileStore/User/changhyuck.lee@novelis.com/recycle_sig_mon/data']
            # Execute the command
            result = subprocess.run(command, capture_output=True, text=True)
        logger.info("Plot has sent successfully")
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