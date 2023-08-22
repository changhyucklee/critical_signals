#How to create a list of object in Python class
#https://www.geeksforgeeks.org/how-to-create-a-list-of-object-in-python-class/
class SigMonData():
    def __init__(self, name,Cols,Stmt,Data):#Signal,Stmt,Plots,Date
        self._name = name
        self._Cols = Cols
        self._Stmt = Stmt
        self._Data = Data
    def get_Data(self,Cols,Start,End,Type):
        return None



if __name__ == "__main__":
    sigmon = SigMonData('Debaler Bearing (DR/NDR) 온도 Monitoring', ['SH BB_DS_BEARING_TEMP','SH BB_NDS_BEARING_TEMP'],lst_stmt_BB_BEARING, df_data_BB_BEARING)
    print(sigmon)


    # df = get_df_filter(['SH BB_DS_BEARING_TEMP','SH BB_NDS_BEARING_TEMP'], StartDay2, ToDay_0630, df_PI)
    # lst_stmt_BB_BEARING, df_data_BB_BEARING = get_data(['Debaler Bearing (DR/NDR) 온도 Monitoring'], date_Out, avg_value_M2, df)
    # df = get_df_filter(['SH RT_DS_BEARING_TEMP','SH RT_NDS_BEARING_TEMP'], StartDay2, ToDay_0630, df_PI)
    # lst_stmt_RT_BEARING, df_data_RT_BEARING = get_data(['#1 Shredder (DR/NDR) 온도 Monitoring'], date_Out, avg_value_M2, df)
    # df = get_df_filter(['SH HD_DS_BEARING_TEMP','SH HD_NDS_BEARING_TEMP'], StartDay2, ToDay_0630, df_PI)
    # lst_stmt_HD_BEARING, df_data_HD_BEARING = get_data(['#2 Shredder (DR/NDR) 온도 Monitoring'], date_Out, avg_value_M2, df)
    # df = get_df_filter(['M22_Amp_Out', 'M38_Amp_Out'], StartDay2, ToDay_0630, df_PI)
    # lst_stmt_Amp_Out, df_data_Amp_Out = get_data(['M22, M38 Conveyor 전류 Monitoring'], date_Out, avg_value_M2, df)
    #
    # df = get_df_filter(['FIKE_BB DE_EIV1','FIKE_BB DE_EIV2','FIKE_BB DE_EIV3'], StartDay2, ToDay_0630, df_PI)
    # lst_stmt_FIKE_BB, df_data_FIKE_BB = get_data(['Fike system damper Monitoring_ Debaler'], date_Out, avg_value_M2, df)
    # df = get_df_filter(['FIKE_SH1 SH1_EIV1','FIKE_SH1 SH1_EIV2'], StartDay2, ToDay_0630, df_PI)
    # lst_stmt_FIKE_SH1, df_data_FIKE_SH1 = get_data(['Fike system damper Monitoring_ #1 Shredder'], date_Out, avg_value_M2, df)
    # df = get_df_filter(['FIKE_SH2 SH2_EIV1','FIKE_SH2 SH2_EIV2'], StartDay2, ToDay_0630, df_PI)
    # lst_stmt_FIKE_SH2, df_data_FIKE_SH2 = get_data(['Fike system damper Monitoring_ #2 Shredder'], date_Out, avg_value_M2, df)
    #
    # df = get_df_filter(['SH BB_AMPS','SH RT_AMPS','SH HD_AMPS'], StartDay2, ToDay_0630, df_PI)
    # lst_stmt_AMPS, df_data_AMPS = get_data(['Debaler, Shreder1, 2 대한 전류값 Monitoring'], date_Out, avg_value_M2, df)
    #
    # df = get_df_filter(['SH BB_AMPS', 'SH RT_AMPS', 'SH HD_AMPS'], StartDay2, ToDay_0630, df_PI)
    # lst_stmt_AMPS, df_data_AMPS = get_data(['Debaler, Shreder1, 2 대한 전류값 Monitoring'], date_Out, avg_value_M2, df)