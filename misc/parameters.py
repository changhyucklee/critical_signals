username = 'mesapp'
password = 'Me$app123'
conn_string = 'DSN=AsiaMES_PROD;' + ';UID=' + username + ';PWD=' + password

da_username  = 'token'
da_password = 'dapi1f5990903533367408bd33808a3fbf78'
da_conn_string='DSN=DAA_DEV;'+';UID='+da_username+';PWD='+ da_password

#Azure_server = 'daafilelog.database.windows.net'
Azure_server = 'AUEGLBWVDAPDP09'
Azure_username = 'agw_de_admin'
Azure_password = 'N0vel1$1'
Azure_database = 'YJ_DIGITAL_USECASE'
Azure_conn_string = 'DSN=Azure_YJ_AGW;' + 'SERVER=' + Azure_server + ';PORT=1433;DATABASE=' + Azure_database + ';UID=' + Azure_username + ';PWD=' + Azure_password


sql_ubc = """
SELECT SUMDAY
      ,DROPNO
      ,ALLOY
      ,ISNULL(UBC_WGT,0) AS UBC_WGT
      ,ISNULL(C1_WGT,0) AS C1_WGT
      ,ISNULL(C2_WGT,0) AS C2_WGT
      ,ISNULL(C3_WGT,0) AS C3_WGT
      ,ISNULL(HOLD_DROSS,0) AS HOLD_DROSS
      ,ISNULL(DRS_WHITE,0) AS DRS_WHITE
      ,ISNULL(DRS_BLACK,0) AS DRS_BLACK 
      ,ISNULL(SALT,0) AS SALT
      ,ISNULL(LOSS_A,0) AS LOSS_A
      ,ISNULL(LOSS_B,0) AS LOSS_B
      ,ISNULL(LOSS_C,0) AS LOSS_C
      ,ISNULL(LOSS_D,0) AS LOSS_D
      ,ISNULL(LOSS_E,0) AS LOSS_E
      ,ISNULL(LOSS_F,0) AS LOSS_F
FROM RCCS_VIEW_TABLE   WITH (NOLOCK)
WHERE SUMDAY > '2021-01-01'
"""
sql_lng = """
SELECT CONVERT(CHAR(10),CONVERT(DATETIME, SUMDAY),121) AS SUMDAY
      ,RK1_A ,RK1_B
      ,RK2_A ,RK2_B
      ,SM1_A ,SM1_B  
      ,SM2_A ,SM2_B
      ,SM3_A ,SM3_B
      ,TM_A  ,TM_B
      ,TH_A  ,TH_B
      ,DBF_A ,DBF_B
  FROM RCCS_LNGUSAGE WITH (NOLOCK)
  WHERE SUMDAY > '2021-01-01'
"""
#따라, DBF 전단  200이상 300 미만 , DBF 후단  160이상 220 미만 일 경우만 계산되어 DBF HEAD LOSS로 시각화 해주시면 감사하겠습니다.
# 즉, 각각 DBF 전단과 후단에 범위를 지정하여 해당 값이 범위내에 있지 않을 경우 하기표에서 제외 부탁드립니다.
sql_SigMon_RFI = """
SELECT BATCHNO
    , DATEADD(mi, -10,  WIPDTTM_ST) AS START_TIME_EX
    , DATEADD(mi, 10,   WIPDTTM_ED) AS END_TIME_EX
    , SUMDAY     As WORK_DATE 
  FROM WipHistory WITH (NOLOCK) 
 WHERE 1=1  
 AND EQPTID = 'YJ1RCHF1'
 AND SUMDAY between '{}' and '{}' 
 """
sql_SigMon_split_jet = """
SELECT BATCHNO
    , DATEADD(mi, -10,  WIPDTTM_ST) AS START_TIME_EX
    , DATEADD(mi, 30,   WIPDTTM_ED) AS END_TIME_EX
    , SUMDAY     As WORK_DATE 
  FROM WipHistory WITH (NOLOCK) 
 WHERE 1=1  
 AND EQPTID = 'YJ1RCHF1'
 AND SUMDAY between '{}' and '{}' 
 """
sql_SigMon_MES = """
SELECT SUBSTRING(H.BATCHNO,1,6) AS BATCHNO
      ,H.SUMDAY      As WORK_DATE 
      ,H.WIPDTTM_ST  AS START_TIME
      ,DATEADD(mi, -10,   H.WIPDTTM_ST) AS START_TIME_EX
      ,H.WIPDTTM_ED AS END_TIME
      ,DATEADD(mi, 10,   H.WIPDTTM_ED) AS END_TIME_EX
      ,DATEADD(mi, 10,   H.WIPDTTM_ST) AS START_Alpur_head
      ,DATEADD(mi, 20,   H.WIPDTTM_ST) AS END_Alpur_head
      ,D.METALLVL_1       -- Metal Level DBF 전단
      ,D.METALLVL_2       -- Metal Level DBF 후단
            ,case when ((D.METALLVL_1 between 200 and 300) and (D.METALLVL_2 between 160 and 220)) then D.METALLVL_1 - D.METALLVL_2 else -1 end AS DBF_head_loss
      --, D.METALLVL_1 - D.METALLVL_2 AS DBF_head_loss
      ,case when ((D.METALLVL_1 between 200 and 300) and (D.METALLVL_2 between 160 and 220)) and (D.METALLVL_1 - D.METALLVL_2 between 40 and 60) then 0 else 1 end check_DBF_head_loss     
      ,D.RT_1        -- 용탕온도 초기
      ,case when (D.RT_1 between 678 and 683) then 0 else 1 end check_RT_1
      ,D.CT_1       -- 냉각수 온도 초기
      ,case when (D.CT_1 between 28.5 and 31.5) then 0 else 1 end check_CT_1
      ,D.BUTTCURL_1 -- Butt Curl측정 #1 
      ,D.BUTTCURL_2 -- Butt Curl측정 #2 
      ,D.BUTTCURL_3 -- Butt Curl측정 #3 
      ,D.BUTTCURL_4 -- Butt Curl측정 #4 
      ,D.BUTTCURL_5 -- Butt Curl측정 #5 
      ,D.BUTTCURL_6 -- Butt Curl측정 #6 
      ,D.BUTTCURL_7 -- Butt Curl측정 #7 
      ,D.BUTTCURL_8 -- Butt Curl측정 #8 
      ,D.CA_REMOVE_RATE
      ,0 check_CA_REMOVE_RATE
      ,ROUND((D.BUTTCURL_1+BUTTCURL_2+BUTTCURL_3+BUTTCURL_6+BUTTCURL_7+BUTTCURL_8) / 6,1) AS BUTTCURL
      ,case when (ROUND((D.BUTTCURL_1+BUTTCURL_2+BUTTCURL_3+BUTTCURL_6+BUTTCURL_7+BUTTCURL_8) / 6,1) between 35 and 45) then 0 else 1 end check_BUTTCURL
      ,SUBSTRING(H.WORKJO,4,1) AS WORKJO
  FROM WipHistory   H WITH (NOLOCK)  
      LEFT OUTER JOIN PERSON P   WITH (NOLOCK) ON H.USERID_ED = P.USERID
            INNER JOIN LOT             L WITH (NOLOCK) ON H.LOTID  = L.LOTID
            INNER JOIN Equipment        E WITH (NOLOCK) ON E.EQPTID = H.EQPTID
            INNER JOIN WipactHistory    WA WITH (NOLOCK) ON L.RTLOTID_M = WA.LOTID AND WA.ACTID = 'DATACOLLECT_LOT'    
            INNER JOIN 
            (
            select 
                 LOTID
                ,WIPSEQ
                ,case when D.CLCTVAL114='' then 0 else CAST(D.CLCTVAL114 AS FLOAT) end as METALLVL_1
                ,case when D.CLCTVAL115='' then 0 else CAST(D.CLCTVAL115 AS FLOAT) end as METALLVL_2
                ,case when D.CLCTVAL108='' then 0 else CAST(D.CLCTVAL108 AS FLOAT) end as RT_1
                ,case when D.CLCTVAL111='' then 0 else CAST(D.CLCTVAL111 AS FLOAT) end as CT_1
                ,case when D.CLCTVAL130='' then 0 else CAST(D.CLCTVAL130 AS FLOAT) end as BUTTCURL_1
                ,case when D.CLCTVAL131='' then 0 else CAST(D.CLCTVAL131 AS FLOAT) end as BUTTCURL_2
                ,case when D.CLCTVAL132='' then 0 else CAST(D.CLCTVAL132 AS FLOAT) end as BUTTCURL_3
                ,case when D.CLCTVAL133='' then 0 else CAST(D.CLCTVAL133 AS FLOAT) end as BUTTCURL_4
                ,case when D.CLCTVAL134='' then 0 else CAST(D.CLCTVAL134 AS FLOAT) end as BUTTCURL_5
                ,case when D.CLCTVAL135='' then 0 else CAST(D.CLCTVAL135 AS FLOAT) end as BUTTCURL_6
                ,case when D.CLCTVAL136='' then 0 else CAST(D.CLCTVAL136 AS FLOAT) end as BUTTCURL_7
                ,case when D.CLCTVAL171='' then 0 else CAST(D.CLCTVAL171 AS FLOAT) end as BUTTCURL_8
                ,case when D.CLCTVAL160='' then 0 else CAST(D.CLCTVAL160 AS FLOAT) end as CA_REMOVE_RATE
              from WipDataCollect D  WITH (NOLOCK)        
            ) D
            ON WA.LOTID = D.LOTID AND WA.WIPSEQ = D.WIPSEQ
 WHERE H.SHOPID     =  'YJ1'
   AND H.PROCID     =  'RCCS'
   AND H.SUMDAY between '{}' and '{}'    
"""

sql_SigMon_PI="""
select 
Timestamp
,System_BatchID_dg
,Rod1_PV_dg,Rod2_PV_dg
,SNIF_ALPUR_Rotor1_CL2_PV_dg,SNIF_ALPUR_Rotor2_CL2_PV_dg,SNIF_ALPUR_Rotor3_CL2_PV_dg,SNIF_ALPUR_Rotor4_CL2_PV_dg

,Degasser_CL2_Storage_Pressure_PV_dg
,case when (Degasser_CL2_Storage_Pressure_PV_dg between 2.5 and 3.5) then 0 else 1 end as check_ALPUR_Storage_Pressure
,SNIF_ALPUR_InletCL2GasPressure_dg
,case when (SNIF_ALPUR_InletCL2GasPressure_dg between 2.5 and 3.5) then 0 else 1 end as check_ALPUR_Main_Pressure

,SNIF_ALPUR_HeaterElement_1_ElementPower,SNIF_ALPUR_HeaterElement_2_ElementPower,SNIF_ALPUR_HeaterElement_3_ElementPower,SNIF_ALPUR_HeaterElement_4_ElementPower

,Degasser_CL2_Scale_dg, Degasser_CL2_Scale_Drop_dg
,Trough_MetalLevel_SP_dg
,Trough_Sensor1_MetalLevel_PV_dg,Trough_Sensor2_MetalLevel_PV_dg
,Trough_MetalLevel_SP_dg - Trough_Sensor2_MetalLevel_PV_dg - 64 as Alpur_head_loss
,Trough_Sensor3_MetalLevel_PV_dg

,Filter1_DBF_Pre_Heater_Box_Temp_PV_dg
,case when (Filter1_DBF_Pre_Heater_Box_Temp_PV_dg between 0 and 800) then 0 else 1 end as check_Box_Temp

,WaterDual_CoolingTower_Turbidity_dg,WaterDual_CoolingTower_Conductibity_dg
,WaterDual_CoolingTower_PH_dg,WaterDual_CoolingTower_ORP_dg

from opsentprodg2.yej_recycle_caster
where Timestamp between '{}' and '{}'          
"""
Machine_list=['Decoater_1','Decoater_2']
tag_list_24h=[
'Delac_1 Decoater Day Running Time Utilization 2',
'Delac_2 Decoater Day Running Time Utilization 2',
'Delac_1 Decoater Day Productivity Avg 2',
'Delac_2 Decoater Day Productivity Avg 2'
]
tag_list_10s=[
'Delac_1 WaterFlow_Afterburner',
'Delac_1 WaterFlow_Duct',
'Delac_1 WaterFlow_Kiln',
'Delac_2 WaterFlow_Afterburner',
'Delac_2 WaterFlow_Duct',
'Delac_2 WaterFlow_Kiln',
'Delac_1 Discharge_Airlock_Faulted',
'Delac_2 Discharge_Airlock_Faulted'
]
af_list_10s=[
'Misc|WaterFlow_Afterburner_dg',
'Misc|WaterFlow_Duct_dg',
'Misc|WaterFlow_Kiln_dg',
'Misc|Discharge_Airlock_Fault_dg'
]
af_list_24h=[
'Misc|Decoater_Day_Running_Time_Utilization_2_dg',
'Misc|Decoater_Day_Productivity_Avg_2_dg'
]

tag_list_1=[
'Delac_1 WTCT3',
'Delac_1 System_Pressure_Control_Valve_Feedback',
'Delac_1 Bag_Pressure_Transmitter1',
'Delac_1 Afterburner_O2',
'Delac_1 Afterburner_Temp',
'Boiler Manned blower front differential pressure PIT403',
'Boiler_CARBONATE_TANK_weight',
'Boiler #1 decoater temperature TE301',
'Boiler #2 decoater temperature TE302',
'Boiler main steam line temperature TE303',
'Boiler Waste heat boiler front pressure PIT301',
'Boiler waste heat boiler rear temp TE304',
'Boiler economizer wasts gas rear temp TE305',
'Boiler dry reactor rear temp TE306',
'Boiler dust collector waste gas front pressure PIT402',
'Delac_1 Kiln_Discharge_O2',
'Delac_1 Diverter_Valve_Motor_Control_Signal',
'Boiler waste heat boiler body temp TE201',
'Boiler manned blower suction temp TE401',
'Delac_1 Kiln_Exit_Temperature',
'Delac_1 Kiln_Inlet_Temperature',
'Delac_1 Recirc_Fan_Motor_Requested_Speed',
'Delac_1 System_Pressure_Transmitter_Discharge',
'Delac_1 Conveyor_Feedrate_PV',
'Delac_1 WaterFlow_Duct',
'Delac_2 Conveyor_Feedrate_PV',
'Delac_2 Afterburner_Temp'
]

tag_list_sigMon=[
"DC_3 ROD_PV_TiBorSpeed",
"DC_3 ROD_PV_MiscSpeed",

"Alpur CHLORINE.AI.Flow_Rotor_1",
"Alpur CHLORINE.AI.Flow_Rotor_2",
"Alpur CHLORINE.AI.Flow_Rotor_3",
"Alpur CHLORINE.AI.Flow_Rotor_4",
"DC_3 B_FurnTiltBackLtch",

"CT Cl2_Storage_Cl2_Pressure",
"Alpur Cl_Main_Pressure",

"Alpur TM.Heater1.Power_Mes",
"Alpur TM.Heater2.Power_Mes",
"Alpur TM.Heater3.Power_Mes",
"Alpur TM.Heater4.Power_Mes",
"Alpur DI_Lid_Closed",

"Cl_Scale_Usage_Drop",
"Cl_Scale",

"DC_3 TGH_SPO_LevelLaser1",
"DC_3 TGH_PV_LevelLaser1",
"DC_3 TGH_PV_LevelLaser2",
"DC_3 TGH_PV_LevelLaser3",

'Nalco8_Turbidity',
'Nalco5_Conductivity',
'Nalco3_pH',
'Nalco4_ORP',
'DBF_Pree TC_BoxTemp',
'DBF_Pree HMI_CoverTemp',
'Delac_1 Conveyor_Feedrate_PV',
'Delac_2 Conveyor_Feedrate_PV',
    
'Delac_1 Bag_Pressure_Transmitter PIT111',
'Delac_1 System_Pressure_Valve_Motor_Control_Sig',
'Delac_1 System_Pressure_Control_Valve_Feedback',
'Delac_2 Bag_Pressure_Transmitter PIT121',
'Delac_2 System_Pressure_Valve_Motor_Control_Sig',
'Delac_2 System_Pressure_Control_Valve_Feedback',   
'Alpur CHLORINE_MAIN.AI.Leak',
'DC_3 PIT_PV_PitWaterLevel',
'CT LIA_201_LT',
'DC_3 CastLengthFromTemposonics',
'CT PCV_202_SV',
'CT PT_201',
'DC_3 WTR_SPO_FaceWaterFlow',
'DC_3 WTR_PV_FaceWaterFlow',
'DC_3 WTR_SPO_EndWaterFlow',
'DC_3 WTR_PV_MoldEndWaterFlow',

'14000_HB AI448',
'14000_HB AI456',
'7000_HB AI336',
'14000_HB AI416',
'4000_HB_BF6_01_PT',
'7000_CB BF3_DPT_DUCT01',
'7000_HB BF2_DPT_DUCT01',
'14000_HB AI352',
'4000_HB_BF6_01_DPT',
'Boiler PIT403-PIT402 _PRESS PIDC402',
'Boiler AOH storage tank level LT601',
'Boiler AE501_PH_Sensor_new',
'Boiler_CARBONATE_TANK_weight',

'Boiler Waste heat boiler front pressure PIT301',
'Boiler dust collector waste gas front pressure PIT402',
'Boiler Manned blower front differential pressure PIT403',

'Delac_1 WTCT3',
'Delac_1 WTCT4',
'Delac_2 WTCT3',
'Delac_2 WTCT4',

'DC3_DC_Cast_No',
'New_RFI_Tilting_Value',
'New_RFI_Salt_Flow_PV',

'DC_3 JET_PV_SplitJetFacePressure',
'SH BB_DS_BEARING_TEMP','SH BB_NDS_BEARING_TEMP',
'SH RT_DS_BEARING_TEMP','SH RT_NDS_BEARING_TEMP',
'SH HD_DS_BEARING_TEMP','SH HD_NDS_BEARING_TEMP',
'M22_Amp_Out', 'M38_Amp_Out',
'FIKE_BB DE_EIV1','FIKE_BB DE_EIV2','FIKE_BB DE_EIV3',
'FIKE_SH1 SH1_EIV1','FIKE_SH1 SH1_EIV2',
'FIKE_SH2 SH2_EIV1','FIKE_SH2 SH2_EIV2',

'SH BB_AMPS','SH RT_AMPS','SH HD_AMPS',

'Delac_1 Cyclone_Inlet_Air_Temp_Setpoint','Delac_1 Cyclone_Inlet_Temp','Delac_1 Kiln_Inlet_Temperature','Delac_1 Recirc_Fan_Motor_Requested_Speed',
'Delac_2 Cyclone_Inlet_Air_Temp_Setpoint','Delac_2 Cyclone_Inlet_Temp','Delac_2 Kiln_Inlet_Temperature','Delac_2 Recirc_Fan_Motor_Requested_Speed'
]
#
#
# today = datetime.date.today()
# StartDay = today - datetime.timedelta(days=8)
# EndDay = today - datetime.timedelta(days=1)
# StartDay_3_1 = today - datetime.timedelta(days=3)
# StartDay_1 = today - datetime.timedelta(days=1)
# StartDay_2 = today - datetime.timedelta(days=2)
#
# MonthStartDay = today - datetime.timedelta(days=31)
# MonthStartDay = MonthStartDay.strftime('%Y-%m-%d')+' 06:30:00'
#
# OneYearAgoDay = today - datetime.timedelta(days=365)
# OneYearAgoDay = OneYearAgoDay.strftime('%Y-%m-%d')+' 06:30:00'
#
# ToDay_0630 = today.strftime('%Y-%m-%d')+' 06:30:00'
#
# StartDay2 = StartDay.strftime('%Y-%m-%d')+' 06:30:00'
# EndDay2 = EndDay.strftime('%Y-%m-%d')+' 06:30:00'
# StartDay_3 = StartDay_3_1.strftime('%Y-%m-%d') + ' 06:30:00'
# StartDay_2_1 = StartDay_2.strftime('%Y-%m-%d') + ' 06:30:00'