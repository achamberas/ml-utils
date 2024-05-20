'''
Set of functions that help with data preprocessing.
'''

import pandas as pd
import numpy as np
import copy

def combineDataset(data_path,
                   input_df_path_list, 
                   output_df_path_list,
                   climate_zones,
                   target_list):

    '''
    Comine multiple datasets together.
    '''

    input_dfs = [pd.read_csv(data_path + input_path) for input_path in input_df_path_list]
    output_dfs = [pd.read_csv(data_path + output_path) for output_path in output_df_path_list]

    result_list = list()

    # Pairwisely preprocess input & output csv files
    for i in range(len(climate_zones)):
        # input_dfs[i]['climate_zone'] = climate_zones[i]

        output_dfs[i] = output_dfs[i].sort_values(by = "sample_id").rename(columns = {'sample_id': 'id'})

        target = output_dfs[i][['id', 'loc'] + [item for item in target_list]]

        result_list.append(pd.merge(input_dfs[i], target, on = "id", how = "inner"))

    final = pd.concat(result_list).reset_index(drop = True).drop(columns = ["id"])

    return final

def getCombinedFeatures(df1, toEnergyUsage = False, target=''):
    """
    This function takes a DataFrame containing building parameters and calculates various building
    characteristics, such as combined floor area (CFA), volume (V), envelope surface area (ESA),
    aspect ratio (AR), total window-to-wall ratio (TWWR), and other relevant metrics. It returns a new
    DataFrame with these calculated metrics added as new columns.

    """

    df = copy.deepcopy(df1)

    LW = df.large_store_width
    SW = df.small_store_width
    D = df.building_depth
    FH = df.ground_floor_height
    DH = 2.13
    LDW = []
    SMD = []
    for i in range(df1.shape[0]):
        # Large Door Width
        if df.ground_floor_elevation_3_wwr[i] < 1.83*2.13*2 / (df.ground_floor_height[i] * df.large_store_width[i]):
            LDW.append(0.91)
        else: LDW.append(1.83)

        # Small Door Width
        if df.ground_floor_elevation_3_wwr[i] < 1.83*2.13 / (df.ground_floor_height[i] * df.small_store_width[i]):
            SMD.append(0.91)
        else: SMD.append(1.83)

    LDW = np.asarray(LDW)
    SDW = np.asarray(SMD)


    df["CFA"] = (2*LW*D) + (8*SW*D)
    df["V"] = df.CFA * FH
    df["OSA"] = (2*D*FH) + (2*FH*((8*SW)+(2*LW)))
    df["ESA"] = df.OSA + (D*((8*SW)+(2*LW)))
    df["ESA:V"] = df.ESA / df.V
    df["AR"] = D / ((2*LW) + (8*SW))
    df["TWWR"]= (FH*((8*SW)+(2*LW))*df.ground_floor_elevation_3_wwr) / df.OSA

    df['WA'] = (2*D*FH) + (FH*((8*SW)+(2*LW))) + ((FH*((8*SW) + (2*LW)))*(1- df.ground_floor_elevation_3_wwr))- ((LDW*DH *4) + (SDW*DH*8))
    df["LS:SS"] = (LW * 2) / (SW * 8)

    df["WWU"] = ((((LW*2) + (SW*8))*FH*df.ground_floor_elevation_3_wwr *df.elv3_window_u_value ) + (df.WA* (1/ df.ext_wall_rsi)) + (((LDW*DH*4) + (SDW*DH*8))*(1/ df.backdoor_rsi))) / df.OSA
    df["OWU"] = ((((LW*2)  + (SW*8))*FH* df.ground_floor_elevation_3_wwr * df.elv3_window_u_value) + (df.WA * (1/ df.ext_wall_rsi)) + (((LW*2)  + (SW*8))*D * (1/df.roof_rsi)) + (((LDW*DH*4) + (SDW*DH*8))*(1/ df.backdoor_rsi))) / df.ESA

    df["ODB"] = df.temp_setpoint_cooling_occupied - df.temp_setpoint_heating_occupied
    df["CD"] = df.temp_setpoint_cooling_setback + df.temp_setpoint_cooling_occupied
    df["HD"] = df.temp_setpoint_heating_occupied - df.temp_setpoint_heating_setback
    df["TLDB"] = df.ground_allowance_lpd + df.ground_lpd

    df["OD"] = ((df.large_max_occupant_density*2)+(df.small_max_occupant_density*8)) / df.CFA

    cols = list(df.columns.values)
    df = df[cols[cols.index("CFA"):] + cols[:cols.index("CFA")]]

    if toEnergyUsage and target != '':
      df[target] = df[target] * df.CFA

    return df


def addLocation(input_df_path_list,
                   climate_zones):
    '''
    Adds location to the dataset 
    '''

    input_dfs = [pd.read_csv(input_path) for input_path in input_df_path_list]

    for i in range(len(climate_zones)):
          # input_dfs[i]['climate_zone'] = climate_zones[i]
          if climate_zones[i] in ['1A', '1B', '2A']:
              input_dfs[i] = input_dfs[i].rename(columns = {'roof_solar_absorbtance': 'roof_solar_absorptance',
                                                  'roof_thermal_absorbtance': 'roof_thermal_absorptance',
                                                  'outside_air_rate': 'outdoor_air_flow_per_area',
                                                  'equip_power_density':'equipment_power_density'
                                                  })
          input_dfs[i] = input_dfs[i].drop(columns = ['id'])
          input_dfs[i]['loc'] = [climate_zones[i]] * len(input_dfs[i])

    final = pd.concat(input_dfs).reset_index(drop = True)

    return final

#@title add climate zone for new test samples (** for now in test process)
def addClimateZone(current_loc, all_loc, values, df):
  """
  Adds climate zones to the dataframe
  
  This function is written to mimic the one hot encoding process
  df: must be the x_train or x_test
  """
  df = df.drop(columns = [current_loc])

  df1 = pd.DataFrame([values] * len(df),
                   columns= ["loc_5A", "loc_5B", "loc_5C", "loc_6A", "loc_6B"])

  df = pd.concat([df, df1], axis = 1)
  return df