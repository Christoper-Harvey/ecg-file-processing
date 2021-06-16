import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import glob
from scipy.signal import medfilt
from scipy.integrate import trapz
import xml.etree.ElementTree as et 
from datetime import date
today = date.today()

np.warnings.filterwarnings('ignore')
sns.set(style="darkgrid")

roots = []
root_names = []
for n in glob.glob('*.xml'):
    roots.append(et.parse(n).getroot())
    root_names.append(n)


def modified_z_score(intensity):
    median_int = np.median(intensity)
    mad_int = np.median([np.abs(intensity - median_int)])
    if mad_int == 0:
        mad_int = 1
    modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
    return modified_z_scores


def df_fixer(y,n):
    
    threshold = 0
    x = 0
    while threshold == 0:
        
        if np.nanquantile(abs(np.array(modified_z_score(np.diff(y)))), 1) > 150:
            if abs(np.array(modified_z_score(np.diff(y))))[int(data.Qonset[n*12])+x:int(data.Qoffset[n*12])+30].max() < np.nanquantile(abs(np.array(modified_z_score(np.diff(y)))), .98)+55:
                threshold = abs(np.array(modified_z_score(np.diff(y))))[int(data.Qonset[n*12])+x:int(data.Qoffset[n*12])+30].max() + 1
            elif abs(np.array(modified_z_score(np.diff(y))))[int(data.Qonset[n*12])+x:int(data.Qoffset[n*12])+30].max() > np.nanquantile(abs(np.array(modified_z_score(np.diff(y)))), .98)+55:
                x += 5
                
                
        elif np.nanquantile(abs(np.array(modified_z_score(np.diff(y)))), 1) <= 150:
            if abs(np.array(modified_z_score(np.diff(y))))[int(data.Qonset[n*12])+x:int(data.Qoffset[n*12])+30].max() < np.nanquantile(abs(np.array(modified_z_score(np.diff(y)))), .992)+55:
                threshold = abs(np.array(modified_z_score(np.diff(y))))[int(data.Qonset[n*12])+x:int(data.Qoffset[n*12])+30].max() + 1
            elif abs(np.array(modified_z_score(np.diff(y))))[int(data.Qonset[n*12])+x:int(data.Qoffset[n*12])+30].max() > np.nanquantile(abs(np.array(modified_z_score(np.diff(y)))), .992)+55:
                x += 5

    spikes = abs(np.array(modified_z_score(np.diff(y)))) > threshold
    y_out = y.copy()
    for i in np.arange(len(spikes)):
        if spikes[i] != 0:
            y_out[i+y_out.index[0]] = None

    return y_out


def half_df_fixer(y,n):
    
    threshold = 0
    x = 0
    while threshold == 0:
        
        if np.nanquantile(abs(np.array(modified_z_score(np.diff(y)))), 1) > 150:
            if abs(np.array(modified_z_score(np.diff(y))))[int(half_data.Qonset[n*12])+x:int(half_data.Qoffset[n*12])+30].max() < np.nanquantile(abs(np.array(modified_z_score(np.diff(y)))), .98)+60:
                threshold = abs(np.array(modified_z_score(np.diff(y))))[int(half_data.Qonset[n*12])+x:int(half_data.Qoffset[n*12])+30].max() + 1
            elif abs(np.array(modified_z_score(np.diff(y))))[int(half_data.Qonset[n*12])+x:int(half_data.Qoffset[n*12])+30].max() > np.nanquantile(abs(np.array(modified_z_score(np.diff(y)))), .98)+60:
                x += 2
    
        elif np.nanquantile(abs(np.array(modified_z_score(np.diff(y)))), 1) <= 150:  
            if abs(np.array(modified_z_score(np.diff(y))))[int(half_data.Qonset[n*12])+x:int(half_data.Qoffset[n*12])+30].max() < np.nanquantile(abs(np.array(modified_z_score(np.diff(y)))), .992)+60:
                threshold = abs(np.array(modified_z_score(np.diff(y))))[int(half_data.Qonset[n*12])+x:int(half_data.Qoffset[n*12])+30].max() + 1
            elif abs(np.array(modified_z_score(np.diff(y))))[int(half_data.Qonset[n*12])+x:int(half_data.Qoffset[n*12])+30].max() > np.nanquantile(abs(np.array(modified_z_score(np.diff(y)))), .992)+60:
                x += 2

    spikes = abs(np.array(modified_z_score(np.diff(y)))) > threshold
    y_out = y.copy()
    for i in np.arange(len(spikes)):
        if spikes[i] != 0:
            y_out[i+y_out.index[0]] = None
            
    return y_out


def hanging_line(point1, point2):

    a = (point2[1] - point1[1])/(np.cosh(point2[0] % 600) - np.cosh(point1[0] % 600))
    b = point1[1] - a*np.cosh(point1[0] % 600)
    x = np.linspace(point1[0], point2[0], (point2[0] - point1[0])+1)
    y = a*np.cosh(x % 600) + b

    return (x,y)


Tags = {'tags':[]}
tags = {'tags':[]}
for root in roots:
    
    if len(root.find('{http://www3.medical.philips.com}waveforms').getchildren()) == 2:
        if int(root.find('{http://www3.medical.philips.com}waveforms')[1].attrib['samplespersec']) == 1000:
            for elem in root.find('{http://www3.medical.philips.com}waveforms')[1]:

                tag = {}
                tag['Lead'] = elem.attrib['leadname']
                if (root[6][1][0][14].text == 'Invalid' or elem[0].text == 'Invalid') and root[6].tag == '{http://www3.medical.philips.com}internalmeasurements':
                    if root[6][1][0][14].text == None or root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text == 'Invalid' or root[6][1][0][14].text == '\n        ' or root[6][1][0][14].text == 'Failed':
                        tag['Ponset'] = 0
                        tag['Pdur'] = 0
                        tag['Print'] = 0
                        tag['Poffset'] = 0
                    else:
                        tag['Ponset'] = int(root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text) - int(root[6][1][0][14].text)
                        tag['Pdur'] = 0
                        tag['Print'] = int(root[6][1][0][14].text)
                        tag['Poffset'] = (int(root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text) - int(root[6][1][0][14].text)) + 0

                elif root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text == 'Invalid' or root[6][1][0][14].text == None or root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text == 'Failed' or root[6][1][0][14].text == 'Failed' or (root[6][1][0][14].text == 'Invalid' or elem[0].text == 'Invalid'):
                    tag['Ponset'] = 0
                    tag['Pdur'] = 0
                    tag['Print'] = 0
                    tag['Poffset'] = 0
                else:
                    tag['Ponset'] = int(root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text) - int(root[6][1][0][14].text)
                    tag['Pdur'] = int(elem[0].text)
                    tag['Print'] = int(root[6][1][0][14].text)
                    tag['Poffset'] = (int(root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text) - int(root[6][1][0][14].text)) + int(elem[0].text)

                if (root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text == 'Invalid' or root[6][0][29].text == 'Invalid' or elem[4].text == 'Invalid' or root[6][1][0][18].text == 'Invalid'):
                    tag['Qonset'] = np.nan
                    tag['Qrsdur'] = np.nan
                    tag['Qoffset'] = np.nan
                    tag['Tonset'] = np.nan
                    tag['Qtint'] = np.nan
                    tag['Toffset'] = np.nan
                    tag['Tdur'] = np.nan
                else:
                    tag['Qonset'] = int(root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text)
                    tag['Qrsdur'] = int(root[6][0][29].text)
                    tag['Qoffset'] =  tag['Qonset'] +  tag['Qrsdur']
                    tag['Tonset'] = int(elem[4].text)
                    tag['Qtint'] = int(root[6][1][0][18].text)
                    tag['Toffset'] =  tag['Qonset'] + tag['Qtint']
                    tag['Tdur'] = tag['Qoffset'] - tag['Qonset']

                if root[7].tag == '{http://www3.medical.philips.com}interpretations' and root[6].tag == '{http://www3.medical.philips.com}internalmeasurements':
                    if root[7][0][1][0].text != None and (root[7][0][1][0].text).isdigit(): tag['HeartRate'] = int(root[7][0][1][0].text)
                    if root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[1].text != None: tag['RRint'] = int(root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[1].text)
                    if root[6][1][0][9].text != None: tag['AtrialRate'] = int(root[6][1][0][9].text)
                    if root[6][0][15].text != None and root[6][0][15].text != 'Indeterminate': tag['QRSFrontAxis'] = int(root[6][0][15].text)
                    if root[6][0][31].text != None and root[6][0][31].text != 'Failed': tag['QTC'] = int(root[6][0][31].text)
                    tag['Target'] = []
                    for n in range(len(root[7][0][root[7][0].getchildren().index(root[7][0].find('{http://www3.medical.philips.com}statement')):])):
                        tag['Target'].append(root[7][0][root[7][0].getchildren().index(root[7][0].find('{http://www3.medical.philips.com}statement')):][n][0].text)
                else:
                    tag['HeartRate'] = np.nan
                    tag['RRint'] = np.nan
                    tag['AtrialRate'] = np.nan
                    tag['QRSFrontAxis'] = np.nan
                    tag['QTC'] = np.nan
                    tag['Target'] = []
                if root[3].tag == '{http://www3.medical.philips.com}reportinfo' and root[5].tag == '{http://www3.medical.philips.com}patient':
                    time = root[3].attrib
                    tag['Date'] = time['date']
                    tag['Time'] = time['time']
                    tag['Sex'] = root[5][0][6].text
                    tag['ID'] = root[5][0][0].text
                    tag['Name'] = root[5][0].find('{http://www3.medical.philips.com}name')[0].text + ', ' + root[5][0].find('{http://www3.medical.philips.com}name')[1].text
                    if root[5][0].find('{http://www3.medical.philips.com}age')[0].tag == '{http://www3.medical.philips.com}dateofbirth':
                        tag['Age'] = int(today.strftime("%Y")) - int(root[5][0].find('{http://www3.medical.philips.com}age')[0].text[0:4])
                    if root[5][0].find('{http://www3.medical.philips.com}age')[0].tag == '{http://www3.medical.philips.com}years':
                        tag['Age'] = int(root[5][0].find('{http://www3.medical.philips.com}age')[0].text)
                tag['Waveform'] = elem[6].text
    #             tag['LongWaveform'] = root[8][0].text
                tags['tags'].append(tag)

        else:
            for elem in root.find('{http://www3.medical.philips.com}waveforms')[1]:

                Tag = {}
                Tag['Lead'] = elem.attrib['leadname']
                if (root[6][1][0][14].text == 'Invalid' or elem[0].text == 'Invalid') and root[6].tag == '{http://www3.medical.philips.com}internalmeasurements':
                    if root[6][1][0][14].text == None or root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text == 'Invalid' or root[6][1][0][14].text == '\n        ' or root[6][1][0][14].text == 'Failed':
                        Tag['Ponset'] = 0
                        Tag['Pdur'] = 0
                        Tag['Print'] = 0
                        Tag['Poffset'] = 0
                    else:
                        Tag['Ponset'] = float(root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text) - int(root[6][1][0][14].text)
                        Tag['Pdur'] = 0
                        Tag['Print'] = int(root[6][1][0][14].text)
                        Tag['Poffset'] = (int(root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text) - int(root[6][1][0][14].text)) + 0

                elif root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text == 'Invalid' or root[6][1][0][14].text == None or root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text == None or root[6][1][0][14].text == 'Invalid' or elem[0].text == 'Invalid' and root[6].tag == '{http://www3.medical.philips.com}internalmeasurements':
                    Tag['Ponset'] = 0
                    Tag['Pdur'] = 0
                    Tag['Print'] = 0
                    Tag['Poffset'] = 0
                else:
                    Tag['Ponset'] = int(root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text) - int(root[6][1][0][14].text)
                    Tag['Pdur'] = int(elem[0].text)
                    Tag['Print'] = int(root[6][1][0][14].text)
                    Tag['Poffset'] = (int(root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text) - int(root[6][1][0][14].text)) + int(elem[0].text)

                if (root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text == 'Invalid' or root[6][1][0][18].text == None or root[6][0][29].text == 'Invalid' or elem[4].text == 'Invalid' or root[6][1][0][18].text == 'Invalid'):
                    Tag['Qonset'] = np.nan
                    Tag['Qrsdur'] = np.nan
                    Tag['Qoffset'] = np.nan
                    Tag['Tonset'] = np.nan
                    Tag['Qtint'] = np.nan
                    Tag['Toffset'] = np.nan
                    Tag['Tdur'] = np.nan
                else:
                    Tag['Qonset'] = int(root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[5].text)
                    Tag['Qrsdur'] = int(root[6][0][29].text)
                    Tag['Qoffset'] =  Tag['Qonset'] +  Tag['Qrsdur']
                    Tag['Tonset'] = int(elem[4].text)
                    Tag['Qtint'] = int(root[6][1][0][18].text)
                    Tag['Toffset'] =  Tag['Qonset'] + Tag['Qtint']
                    Tag['Tdur'] = Tag['Qoffset'] - Tag['Qonset']

                if root[7].tag == '{http://www3.medical.philips.com}interpretations' and root[6].tag == '{http://www3.medical.philips.com}internalmeasurements':
                    if root[7][0][1][0].text != None and (root[7][0][1][0].text).isdigit(): Tag['HeartRate'] = int(root[7][0][1][0].text)
                    if root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[1].text != None: Tag['RRint'] = int(root[7][0].find('{http://www3.medical.philips.com}globalmeasurements')[1].text)
                    if root[6][1][0][9].text != None: Tag['AtrialRate'] = int(root[6][1][0][9].text)
                    if root[6][0][15].text != None and root[6][0][15].text != 'Indeterminate': Tag['QRSFrontAxis'] = int(root[6][0][15].text)
                    if root[6][0][31].text != None: Tag['QTC'] = int(root[6][0][31].text)
                    Tag['Target'] = []
                    for n in range(len(root[7][0][root[7][0].getchildren().index(root[7][0].find('{http://www3.medical.philips.com}statement')):])):
                        Tag['Target'].append(root[7][0][root[7][0].getchildren().index(root[7][0].find('{http://www3.medical.philips.com}statement')):][n][0].text)

                else:
                    Tag['HeartRate'] = np.nan
                    Tag['RRint'] = np.nan
                    Tag['AtrialRate'] = np.nan
                    Tag['QRSFrontAxis'] = np.nan
                    Tag['QTC'] = np.nan
                    Tag['Target'] = []
                if root[3].tag == '{http://www3.medical.philips.com}reportinfo' and root[5].tag == '{http://www3.medical.philips.com}patient':
                    time = root[3].attrib
                    Tag['Date'] = time['date']
                    Tag['Time'] = time['time']
                    Tag['Sex'] = root[5][0][6].text
                    Tag['ID'] = root[5][0][0].text
                    Tag['Name'] = root[5][0].find('{http://www3.medical.philips.com}name')[0].text + ', ' + root[5][0].find('{http://www3.medical.philips.com}name')[1].text
                    if len(root[5][0].find('{http://www3.medical.philips.com}age')) > 0:
                        if root[5][0].find('{http://www3.medical.philips.com}age')[0].tag == '{http://www3.medical.philips.com}dateofbirth':
                            Tag['Age'] = int(today.strftime("%Y")) - int(root[5][0].find('{http://www3.medical.philips.com}age')[0].text[0:4])
                        if root[5][0].find('{http://www3.medical.philips.com}age')[0].tag == '{http://www3.medical.philips.com}years':
                            Tag['Age'] = int(root[5][0].find('{http://www3.medical.philips.com}age')[0].text)
                Tag['Waveform'] = elem[6].text
    #             Tag['LongWaveform'] = root[8][0].text
                Tags['tags'].append(Tag)
            
half_data = pd.DataFrame(Tags['tags'])
data = pd.DataFrame(tags['tags'])

del roots
del root
del elem

count1000 = int(len(data)/12)
count500 = int(len(half_data)/12)
count = count1000 + count500

if len(data) > 0:
    array = np.unique(data[data.isnull().any(axis=1)][['ID', 'Date', 'Time']])
    missing_data = data.loc[data['ID'].isin(array) & data['Date'].isin(array) & data['Time'].isin(array)]
    data.drop(missing_data.index, axis=0,inplace=True)
    missing_data = missing_data.reset_index(drop=True)
    del tag
    del tags
    
    data = data.reset_index(drop=True)
    
    for n in range(count1000):
        data.Tonset[n*12:(n+1)*12] = np.repeat(int(data.Tonset[n*12:(n+1)*12].sum()/12), 12)
        data.Pdur[n*12:(n+1)*12] = np.repeat(int(data.Pdur[n*12:(n+1)*12].sum()/12), 12)
    
    
    x = 0
    p = []
    for x in range(len(data.Waveform)):
        t = base64.b64decode(data.Waveform[x])
        p.append(np.asarray(t))
        x+=1
    p = np.asarray(p)
    a = []

    for i in p:
        o = []
        for x in i:
            o.append(x)
        a.append(o)

    df = pd.DataFrame(a)
    df.insert(0, 'Lead', data['Lead'])

    blank = []
    for n in range(count1000):
        blank.append(pd.pivot_table(df[(n*12):(n+1)*12], columns=df.Lead))
        test = pd.concat(blank)

    new = []
    array = []

    for n in range(13):
        for index, num in zip(test.iloc[:, n-1][::2], test.iloc[:, n-1][1::2]):
            if num > 128:
                    new.append(index - (256 * (256 - num)))
            elif num < 128:
                    new.append(index + (256 * num))
            elif num == 0:
                    new.append(index)
            else:
                new.append(index)
        new = []
        array.append(new)

    array = np.asarray([array[0], array[1], array[2], array[3], array[4], array[5], array[6], array[7], array[8], array[9], array[10], array[11]])
    df = pd.DataFrame(array)
    df = pd.pivot_table(df, columns=test.columns)
    df = df.fillna(0)

    del a
    del p
    del o
    del t
    del blank
    del new
    del array

    for n in range(count1000):
        for x in range(12):

            if (data.Toffset[n*12]-data.RRint[n*12]) >= data.Ponset[n*12] or (data.Ponset[n*12] + data.RRint[n*12]) - data.Toffset[n*12] == 1:
                df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - (df.iloc[:,x][n*1200:int(data.Qonset[n*12])+(n*1200)].mean() + df.iloc[:,x][int(data.Qoffset[n*12])+(n*1200):(n+1)*1200].mean()) / 2

            else:
                rrint = data.RRint[n*12]
                if (rrint + data.Ponset[n*12]) > 1200 and (data.Toffset[n*12]-rrint) < 0:

                    temp = df.iloc[:,x][int(n*1200):int(data.Ponset[n*12]+(n*1200))]
                    test = df.iloc[:,x][int(data.Toffset[n*12]+(n*1200)):int((n+1)*1200)]

                    if test.empty == False and temp.empty == False:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - ((temp[len(temp)//3:len(temp)*2//3].mean() + test[len(test)//3:len(test)*2//3].mean()) / 2)
                    elif temp.empty:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - test[len(test)//3:len(test)*2//3].mean()
                    elif test.empty:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - temp[len(temp)//3:len(temp)*2//3].mean()
                    elif test.empty and temp.empty:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - (df.iloc[:,x][n*1200:int(data.Qonset[n*12])+(n*1200)].mean() + df.iloc[:,x][int(data.Qoffset[n*12])+(n*1200):(n+1)*1200].mean()) / 2


                elif (rrint + data.Ponset[n*12]) > 1200 and (data.Toffset[n*12]-rrint) > 0:

                    temp = df.iloc[:,x][int(data.Toffset[n*12]+(n*1200)-rrint):int(data.Ponset[n*12]+(n*1200))]
                    test = df.iloc[:,x][int(data.Toffset[n*12]+(n*1200)):int((n+1)*1200)]

                    if test.empty == False and temp.empty == False:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - ((temp[len(temp)//3:len(temp)*2//3].mean() + test[len(test)//3:len(test)*2//3].mean()) / 2)
                    elif temp.empty:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - test[len(test)//3:len(test)*2//3].mean()
                    elif test.empty:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - temp[len(temp)//3:len(temp)*2//3].mean()
                    elif test.empty and temp.empty:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - (df.iloc[:,x][n*1200:int(data.Qonset[n*12])+(n*1200)].mean() + df.iloc[:,x][int(data.Qoffset[n*12])+(n*1200):(n+1)*1200].mean()) / 2


                elif rrint + data.Ponset[n*12] < 1200 and (data.Toffset[n*12]-rrint) < 0:

                    temp = df.iloc[:,x][int(n*1200):int(data.Ponset[n*12]+(n*1200))]
                    test = df.iloc[:,x][int(data.Toffset[n*12]+(n*1200)):int(rrint + data.Ponset[n*12]+(n*1200))]

                    if test.empty == False and temp.empty == False:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - ((temp[len(temp)//3:len(temp)*2//3].mean() + test[len(test)//3:len(test)*2//3].mean()) / 2)
                    elif temp.empty:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - test[len(test)//3:len(test)*2//3].mean()
                    elif test.empty:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - temp[len(temp)//3:len(temp)*2//3].mean()
                    elif test.empty and temp.empty:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - (df.iloc[:,x][n*1200:int(data.Qonset[n*12])+(n*1200)].mean() + df.iloc[:,x][int(data.Qoffset[n*12])+(n*1200):(n+1)*1200].mean()) / 2

                else:

                    temp = df.iloc[:,x][int(data.Toffset[n*12]+(n*1200)-rrint):int(data.Ponset[n*12]+(n*1200))]
                    test = df.iloc[:,x][int(data.Toffset[n*12]+(n*1200)):int(rrint + data.Ponset[n*12]+(n*1200))]

                    if test.empty == False and temp.empty == False:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - ((temp[len(temp)//3:len(temp)*2//3].mean() + test[len(test)//3:len(test)*2//3].mean()) / 2)
                    elif temp.empty:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - test[len(test)//3:len(test)*2//3].mean()
                    elif test.empty:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - temp[len(temp)//3:len(temp)*2//3].mean()
                    elif test.empty and temp.empty:
                        df.iloc[:,x][n*1200:1200*(n+1)] = df.iloc[:,x][n*1200:1200*(n+1)] - (df.iloc[:,x][n*1200:int(data.Qonset[n*12])+(n*1200)].mean() + df.iloc[:,x][int(data.Qoffset[n*12])+(n*1200):(n+1)*1200].mean()) / 2

    unfiltered_leads = df.copy()
    
    for n in range(count1000):
        for inx in range(12):

            test = df_fixer(df.iloc[:,inx][n*1200:(n+1)*1200], n)

            gaps = []
            lstOfNs = []
            gap = []

            for num in test[test.isna() == True].index:

                lstOfNs.append(num)
                if len(lstOfNs) == 1:
                    gap.append(lstOfNs[0])

                if len(lstOfNs) > 1:

                    if lstOfNs[-1] - lstOfNs[-2] < 5:
                        gap.append(num)
                    elif lstOfNs[-1] - lstOfNs[-2] > 5: 
                        gaps.append(gap)
                        gap = []
                        gap.append(num)

            gaps.append(gap)

            if gaps != [[]]:
                x = []
                y = []
                for g in gaps:

                    if len(g) == 1:

                        x.append([g[-1]+1])
                        y.append(test[g[-1]+1])

                    if np.isnan(test.iloc[0]):

                        point1 = [g[0], test[g[-1]+1]]
                        point2 = [g[-1]+1, test[g[-1]+1]]
                        x_temp,y_temp = hanging_line(point1, point2)
                        x.append(x_temp)
                        y.append(y_temp)

                    else:

                        point1 = [g[0]-1, test[g[0]-1]]
                        point2 = [g[-1]+1, test[g[-1]+1]]
                        x_temp,y_temp = hanging_line(point1, point2)
                        x.append(x_temp)
                        y.append(y_temp)

                for i in range(len(x)):
                    test[x[i]] = y[i]


            if (trapz(abs(test[int(data.Qonset[n*12]):int(data.Qoffset[n*12])]))/trapz(abs(df.iloc[:,inx][int(data.Qonset[12*n]+(1200*n)):int(data.Qoffset[12*n]+(1200*n))]))) < .60:

                test = df.iloc[:,inx][n*1200:(n+1)*1200]

            test = medfilt(test, kernel_size=9)  

            df.iloc[:,inx][n*1200:(n+1)*1200] = test
    
    del gaps
    del lstOfNs
    del gap
    del test
    
    VTI_leads = df[['III', 'aVF', 'aVL', 'aVR']]
    df = df[['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']]
    Unfiltered_VTI_leads = unfiltered_leads[['III', 'aVF', 'aVL', 'aVR']]
    unfiltered_leads = unfiltered_leads[['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']]
    
    matrix = [[.38, -.07, -.13, .05, -.01, .14, .06, .54],
              [-.07, .93, .06, -.02, -.05, .06, -.17, .13],
              [.11, -.23, -.43, -.06, -.14, -.20, -.11, .31]]

    x = matrix[0]
    y = matrix[1]
    z = matrix[2]

    n = 0
    xtemp = []
    ytemp = []
    ztemp = []
    for i in range(len(df)):

        xtemp.append((df.iloc[n].values * x).sum())
        ytemp.append((df.iloc[n].values * y).sum())
        ztemp.append((df.iloc[n].values * z).sum())
        n+=1

    df['x'] = xtemp
    df['y'] = ytemp
    df['z'] = ztemp

    n = 0
    xtemp = []
    ytemp = []
    ztemp = []
    for i in range(len(unfiltered_leads)):

        xtemp.append((unfiltered_leads.iloc[n].values * x).sum())
        ytemp.append((unfiltered_leads.iloc[n].values * y).sum())
        ztemp.append((unfiltered_leads.iloc[n].values * z).sum())
        n+=1

    df['Unfiltered_x'] = xtemp
    df['Unfiltered_y'] = ytemp
    df['Unfiltered_z'] = ztemp
    
    del xtemp
    del ytemp
    del ztemp
    
    
    df['Date'] = data['Date']
    df['ID'] = data['ID']
    df['Time'] = data['Time']
    df['Print'] = data['Print']
    df['Ponset'] = data['Ponset']
    df['Pdur'] = data['Pdur']
    df['Poffset'] = data['Poffset']
    df['Qonset'] = data['Qonset']
    df['Qrsdur'] = data['Qrsdur']
    df['Qtint'] = data['Qtint']
    df['Qoffset'] = data['Qoffset']
    df['Tonset'] = data['Tonset']
    df['Tdur'] = data['Tdur']
    df['Toffset'] = data['Toffset']
    df['HeartRate'] = data['HeartRate']
    df['QRSFrontAxis'] = data['QRSFrontAxis']
    df['Sex'] = data['Sex']
    df['QTC'] = data['QTC']
    df['Age'] = data['Age']
    df['Name'] = data['Name']

    for n in range(count1000):
        df['Ponset'][(n*1200):(n+1)*1200] = data['Ponset'][n*12]
        df['Print'][(n*1200):(n+1)*1200] = data['Print'][n*12]
        df['Pdur'][(n*1200):(n+1)*1200] = data['Pdur'][n*12]
        df['Poffset'][(n*1200):(n+1)*1200] = data['Poffset'][n*12]
        df['Qonset'][(n*1200):(n+1)*1200] = data['Qonset'][n*12]
        df['Qrsdur'][(n*1200):(n+1)*1200] = data['Qrsdur'][n*12]
        df['Qtint'][(n*1200):(n+1)*1200] = data['Qtint'][n*12]
        df['Qoffset'][(n*1200):(n+1)*1200] = data['Qoffset'][n*12]
        df['Tonset'][(n*1200):(n+1)*1200] = data['Tonset'][n*12]
        df['Tdur'][(n*1200):(n+1)*1200] = data['Tdur'][n*12]
        df['Toffset'][(n*1200):(n+1)*1200] = data['Toffset'][n*12]
        df['HeartRate'][(n*1200):(n+1)*1200] = data['HeartRate'][n*12]
        df['QRSFrontAxis'][(n*1200):(n+1)*1200] = data['QRSFrontAxis'][n*12]
        df['Sex'][(n*1200):(n+1)*1200] = data['Sex'][n*12]
        df['QTC'][(n*1200):(n+1)*1200] = data['QTC'][n*12]
        df['Age'][(n*1200):(n+1)*1200] = data['Age'][n*12]
        df['Date'][(n*1200):(n+1)*1200] = data['Date'][12*n]
        df['Time'][(n*1200):(n+1)*1200] = data['Time'][12*n]
        df['ID'][(n*1200):(n+1)*1200] = data['ID'][12*n]
        df['Name'][(n*1200):(n+1)*1200] = data['Name'][12*n]
    
    df[['III', 'aVF', 'aVL', 'aVR']] = VTI_leads
    unfiltered_leads[['III', 'aVF', 'aVL', 'aVR']] = Unfiltered_VTI_leads
    df[['Unfiltered_I', 'Unfiltered_II', 'Unfiltered_III', 'Unfiltered_V1', 'Unfiltered_V2', 'Unfiltered_V3', 'Unfiltered_V4', 'Unfiltered_V5', 'Unfiltered_V6', 'Unfiltered_aVF', 'Unfiltered_aVL', 'Unfiltered_aVR']] = unfiltered_leads[['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']]

    del unfiltered_leads
    del VTI_leads
    
    
if len(half_data) > 0:
    array = np.unique(half_data[half_data.isnull().any(axis=1)][['ID', 'Date', 'Time']])
    missing_half_data = half_data.loc[half_data['ID'].isin(array) & half_data['Date'].isin(array) & half_data['Time'].isin(array)]
    half_data.drop(missing_half_data.index, axis=0,inplace=True)
    missing_half_data = missing_half_data.reset_index(drop=True)
    del Tag
    del Tags
    
    half_data = half_data.reset_index(drop=True)

    for n in range(count500):
        half_data.Tonset[n*12:(n+1)*12] = np.repeat(int(half_data.Tonset[n*12:(n+1)*12].sum()/12), 12)
        half_data.Pdur[n*12:(n+1)*12] = np.repeat(int(half_data.Pdur[n*12:(n+1)*12].sum()/12), 12)

    x = 0
    p = []
    for x in range(len(half_data.Waveform)):
        t = base64.b64decode(half_data.Waveform[x])
        p.append(np.asarray(t))
        x+=1 
    p = np.asarray(p)
    a = []

    for i in p:
        o = []
        for x in i:
            o.append(x)
        a.append(o)

    half_df = pd.DataFrame(a)
    half_df.insert(0, 'Lead', half_data['Lead'])

    blank = []
    for n in range(count500):
        blank.append(pd.pivot_table(half_df[(n*12):(n+1)*12], columns=half_df.Lead))
        test = pd.concat(blank)

    new = []
    array = []

    for n in range(13):
        for index, num in zip(test.iloc[:, n-1][::2], test.iloc[:, n-1][1::2]):

            if num > 128:
                    new.append(index - (256 * (256 - num)))
            elif num < 128:
                    new.append(index + (256 * num))
            elif num == 0:
                    new.append(index)
            else:
                new.append(index)
        new = []
        array.append(new)

    array = np.asarray([array[0], array[1], array[2], array[3], array[4], array[5], array[6], array[7], array[8], array[9], array[10], array[11]])
    half_df = pd.DataFrame(array)

    half_df = pd.pivot_table(half_df, columns=test.columns)
    half_df = half_df.fillna(0)

    blank = []
    for n in range(count500):
        blank.append(half_df[(n*1200):((n+1)*1200)-600])
        test = pd.concat(blank)
    half_df = test
    half_df = half_df.reset_index(drop=True)
    half_df = pd.pivot_table(half_df, columns=half_df.index)


    array = []
    for i in range(count500):
        for x in range(12):
            temp = []
            new = []

            for n in half_df.iloc[x,i*600:(i+1)*600]:
                temp.append(n)
                if len(temp) > 1:
                    new.append(temp[-2])

                if len(temp) < 601 and len(temp) > 1:
                    new.append((temp[-1]+temp[-2])/2)

                if len(temp) == 600:
                    new.append(temp[-1])
                    new.append(temp[-1])
            array.append(new)

    I = (np.asarray(array[::12])).reshape(count500*1200)
    II = (np.asarray(array[1::12])).reshape(count500*1200)
    III = (np.asarray(array[2::12])).reshape(count500*1200)
    V1 = (np.asarray(array[3::12])).reshape(count500*1200)
    V2 = (np.asarray(array[4::12])).reshape(count500*1200)
    V3 = (np.asarray(array[5::12])).reshape(count500*1200)
    V4 = (np.asarray(array[6::12])).reshape(count500*1200)
    V5 = (np.asarray(array[7::12])).reshape(count500*1200)
    V6 = (np.asarray(array[8::12])).reshape(count500*1200)
    aVF = (np.asarray(array[9::12])).reshape(count500*1200)
    aVL = (np.asarray(array[10::12])).reshape(count500*1200)
    aVR = (np.asarray(array[11::12])).reshape(count500*1200)

    half_df = pd.pivot_table(pd.DataFrame([I, II, III, V1, V2, V3, V4, V5, V6, aVF, aVL, aVR]), columns=test.columns)
    half_df = half_df.fillna(0)
    
    del I
    del II 
    del III
    del V1 
    del V2 
    del V3 
    del V4 
    del V5 
    del V6 
    del aVF
    del aVL
    del aVR
    del a
    del p
    del o
    del t
    del blank
    del new
    del array
    del temp
    
    for n in range(count500):
        for x in range(12):

            if ((half_data.Toffset[n*12]-half_data.RRint[n*12]) >= half_data.Ponset[n*12]) or ((half_data.Ponset[n*12] + half_data.RRint[n*12]) - half_data.Toffset[n*12] == 1):
                    half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - (half_df.iloc[:,x][n*1200:int(half_data.Qonset[n*12])+(n*1200)].mean() + half_df.iloc[:,x][int(half_data.Qoffset[n*12])+(n*1200):(n+1)*1200].mean()) / 2

            else:
                rrint = half_data.RRint[n*12]
                if (rrint + half_data.Ponset[n*12]) > 1200 and (half_data.Toffset[n*12]-rrint) < 0:

                    temp = half_df.iloc[:,x][int(n*1200):int(half_data.Ponset[n*12]+(n*1200))]
                    test = half_df.iloc[:,x][int(half_data.Toffset[n*12]+(n*1200)):int((n+1)*1200)]

                    if test.empty == False and temp.empty == False:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - ((temp[len(temp)//3:len(temp)*2//3].mean() + test[len(test)//3:len(test)*2//3].mean()) / 2)
                    elif temp.empty:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - test[len(test)//3:len(test)*2//3].mean()
                    elif test.empty:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - temp[len(temp)//3:len(temp)*2//3].mean()
                    elif test.empty and temp.empty:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - (half_df.iloc[:,x][n*1200:int(half_data.Qonset[n*12])+(n*1200)].mean() + half_df.iloc[:,x][int(half_data.Qoffset[n*12])+(n*1200):(n+1)*1200].mean()) / 2


                elif (rrint + half_data.Ponset[n*12]) > 1200 and (half_data.Toffset[n*12]-rrint) > 0:

                    temp = half_df.iloc[:,x][int(half_data.Toffset[n*12]+(n*1200)-rrint):int(half_data.Ponset[n*12]+(n*1200))]
                    test = half_df.iloc[:,x][int(half_data.Toffset[n*12]+(n*1200)):int((n+1)*1200)]

                    if test.empty == False and temp.empty == False:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - ((temp[len(temp)//3:len(temp)*2//3].mean() + test[len(test)//3:len(test)*2//3].mean()) / 2)
                    elif temp.empty:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - test[len(test)//3:len(test)*2//3].mean()
                    elif test.empty:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - temp[len(temp)//3:len(temp)*2//3].mean()
                    elif test.empty and temp.empty:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - (half_df.iloc[:,x][n*1200:int(half_data.Qonset[n*12])+(n*1200)].mean() + half_df.iloc[:,x][int(half_data.Qoffset[n*12])+(n*1200):(n+1)*1200].mean()) / 2


                elif rrint + half_data.Ponset[n*12] < 1200 and (half_data.Toffset[n*12]-rrint) < 0:

                    temp = half_df.iloc[:,x][int(n*1200):int(half_data.Ponset[n*12]+(n*1200))]
                    test = half_df.iloc[:,x][int(half_data.Toffset[n*12]+(n*1200)):int(rrint + half_data.Ponset[n*12]+(n*1200))]

                    if test.empty == False and temp.empty == False:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - ((temp[len(temp)//3:len(temp)*2//3].mean() + test[len(test)//3:len(test)*2//3].mean()) / 2)
                    elif temp.empty:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - test[len(test)//3:len(test)*2//3].mean()
                    elif test.empty:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - temp[len(temp)//3:len(temp)*2//3].mean()
                    elif test.empty and temp.empty:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - (half_df.iloc[:,x][n*1200:int(half_data.Qonset[n*12])+(n*1200)].mean() + half_df.iloc[:,x][int(half_data.Qoffset[n*12])+(n*1200):(n+1)*1200].mean()) / 2

                else:

                    temp = half_df.iloc[:,x][int(half_data.Toffset[n*12]+(n*1200)-rrint):int(half_data.Ponset[n*12]+(n*1200))]
                    test = half_df.iloc[:,x][int(half_data.Toffset[n*12]+(n*1200)):int(rrint + half_data.Ponset[n*12]+(n*1200))]

                    if test.empty == False and temp.empty == False:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - ((temp[len(temp)//3:len(temp)*2//3].mean() + test[len(test)//3:len(test)*2//3].mean()) / 2)
                    elif temp.empty:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - test[len(test)//3:len(test)*2//3].mean()
                    elif test.empty:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - temp[len(temp)//3:len(temp)*2//3].mean()
                    elif test.empty and temp.empty:
                        half_df.iloc[:,x][n*1200:1200*(n+1)] = half_df.iloc[:,x][n*1200:1200*(n+1)] - (half_df.iloc[:,x][n*1200:int(half_data.Qonset[n*12])+(n*1200)].mean() + half_df.iloc[:,x][int(half_data.Qoffset[n*12])+(n*1200):(n+1)*1200].mean()) / 2

    for x in range(12): 
        half_df.iloc[:,x] = half_df.iloc[:,x]*2.5
    
    unfiltered_half_leads = half_df.copy()
    
    for n in range(count500):

        for inx in range(12):

            test = half_df_fixer(half_df.iloc[:,inx][n*1200:(n+1)*1200], n)

            gaps = []
            lstOfNs = []
            gap = []

            for num in test[test.isna() == True].index:

                lstOfNs.append(num)
                if len(lstOfNs) == 1:
                    gap.append(lstOfNs[0])

                if len(lstOfNs) > 1:

                    if lstOfNs[-1] - lstOfNs[-2] < 5:
                        gap.append(num)
                    elif lstOfNs[-1] - lstOfNs[-2] > 5: 
                        gaps.append(gap)
                        gap = []
                        gap.append(num)

            gaps.append(gap)

            if gaps != [[]]:
                x = []
                y = []
                for g in gaps:

                    if len(g) == 1:

                        x.append([g[-1]+1])
                        y.append(test[g[-1]+1])

                    if np.isnan(test.iloc[0]):

                        point1 = [g[0], test[g[-1]+1]]
                        point2 = [g[-1]+1, test[g[-1]+1]]
                        x_temp,y_temp = hanging_line(point1, point2)
                        x.append(x_temp)
                        y.append(y_temp)

                    else:

                        point1 = [g[0]-1, test[g[0]-1]]
                        point2 = [g[-1]+1, test[g[-1]+1]]
                        x_temp,y_temp = hanging_line(point1, point2)
                        x.append(x_temp)
                        y.append(y_temp)

                for i in range(len(x)):
                    test[x[i]] = y[i]


            if (trapz(abs(test[int(half_data.Qonset[n*12]):int(half_data.Qoffset[n*12])]))/trapz(abs(half_df.iloc[:,inx][int(half_data.Qonset[12*n]+(1200*n)):int(half_data.Qoffset[12*n]+(1200*n))]))) < .60:

                test = half_df.iloc[:,inx][n*1200:(n+1)*1200]

            test = medfilt(test, kernel_size=9)  

            half_df.iloc[:,inx][n*1200:(n+1)*1200] = test
            
    del gaps
    del lstOfNs
    del gap
    del test
    
    half_VTI_leads = half_df[['III', 'aVF', 'aVL', 'aVR']]
    half_df = half_df[['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']]
    Unfiltered_half_VTI_leads = unfiltered_half_leads[['III', 'aVF', 'aVL', 'aVR']]
    unfiltered_half_leads = unfiltered_half_leads[['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']]

    matrix = [[.38, -.07, -.13, .05, -.01, .14, .06, .54],
              [-.07, .93, .06, -.02, -.05, .06, -.17, .13],
              [.11, -.23, -.43, -.06, -.14, -.20, -.11, .31]]

    x = matrix[0]
    y = matrix[1]
    z = matrix[2]

    n = 0
    xtemp = []
    ytemp = []
    ztemp = []
    for i in range(len(half_df)):

        xtemp.append((half_df.iloc[n].values * x).sum())
        ytemp.append((half_df.iloc[n].values * y).sum())
        ztemp.append((half_df.iloc[n].values * z).sum())
        n+=1

    half_df['x'] = xtemp
    half_df['y'] = ytemp
    half_df['z'] = ztemp


    x = matrix[0]
    y = matrix[1]
    z = matrix[2]


    n = 0
    xtemp = []
    ytemp = []
    ztemp = []
    for i in range(len(unfiltered_half_leads)):

        xtemp.append((unfiltered_half_leads.iloc[n].values * x).sum())
        ytemp.append((unfiltered_half_leads.iloc[n].values * y).sum())
        ztemp.append((unfiltered_half_leads.iloc[n].values * z).sum())
        n+=1

    half_df['Unfiltered_x'] = xtemp
    half_df['Unfiltered_y'] = ytemp
    half_df['Unfiltered_z'] = ztemp

    del xtemp
    del ytemp
    del ztemp
    
    
    half_df['Date'] = half_data['Date']
    half_df['ID'] = half_data['ID']
    half_df['Time'] = half_data['Time']
    half_df['Ponset'] = half_data['Ponset']
    half_df['Print'] = half_data['Print']
    half_df['Pdur'] = half_data['Pdur']
    half_df['Poffset'] = half_data['Poffset']
    half_df['Qonset'] = half_data['Qonset']
    half_df['Qrsdur'] = half_data['Qrsdur']
    half_df['Qtint'] = half_data['Qtint']
    half_df['Qoffset'] = half_data['Qoffset']
    half_df['Tonset'] = half_data['Tonset']
    half_df['Tdur'] = half_data['Tdur']
    half_df['Toffset'] = half_data['Toffset']
    half_df['HeartRate'] = half_data['HeartRate']
    half_df['QRSFrontAxis'] = half_data['QRSFrontAxis']
    half_df['Sex'] = half_data['Sex']
    half_df['QTC'] = half_data['QTC']
    half_df['Age'] = half_data['Age']
    half_df['Name'] = half_data['Name']

    for n in range(count500):
        half_df['Ponset'][(n*1200):(n+1)*1200] = half_data['Ponset'][n*12]
        half_df['Print'][(n*1200):(n+1)*1200] = half_data['Print'][n*12]
        half_df['Pdur'][(n*1200):(n+1)*1200] = half_data['Pdur'][n*12]
        half_df['Poffset'][(n*1200):(n+1)*1200] = half_data['Poffset'][n*12]
        half_df['Qonset'][(n*1200):(n+1)*1200] = half_data['Qonset'][n*12]
        half_df['Qrsdur'][(n*1200):(n+1)*1200] = half_data['Qrsdur'][n*12]
        half_df['Qtint'][(n*1200):(n+1)*1200] = half_data['Qtint'][n*12]
        half_df['Qoffset'][(n*1200):(n+1)*1200] = half_data['Qoffset'][n*12]
        half_df['Tonset'][(n*1200):(n+1)*1200] = half_data['Tonset'][n*12]
        half_df['Tdur'][(n*1200):(n+1)*1200] = half_data['Tdur'][n*12]
        half_df['Toffset'][(n*1200):(n+1)*1200] = half_data['Toffset'][n*12]
        half_df['HeartRate'][(n*1200):(n+1)*1200] = half_data['HeartRate'][n*12]
        half_df['QRSFrontAxis'][(n*1200):(n+1)*1200] = half_data['QRSFrontAxis'][n*12]
        half_df['Sex'][(n*1200):(n+1)*1200] = half_data['Sex'][n*12]
        half_df['QTC'][(n*1200):(n+1)*1200] = half_data['QTC'][n*12]
        half_df['Name'][(n*1200):(n+1)*1200] = half_data['Name'][12*n]
        half_df['Age'][(n*1200):(n+1)*1200] = half_data['Age'][12*n]
        half_df['ID'][(n*1200):(n+1)*1200] = half_data['ID'][12*n]
        half_df['Date'][(n*1200):(n+1)*1200] = half_data['Date'][12*n]
        half_df['Time'][(n*1200):(n+1)*1200] = half_data['Time'][12*n]
    
    half_df[['III', 'aVF', 'aVL', 'aVR']] = half_VTI_leads
    unfiltered_half_leads[['III', 'aVF', 'aVL', 'aVR']] = Unfiltered_half_VTI_leads
    half_df[['Unfiltered_I', 'Unfiltered_II', 'Unfiltered_III', 'Unfiltered_V1', 'Unfiltered_V2', 'Unfiltered_V3', 'Unfiltered_V4', 'Unfiltered_V5', 'Unfiltered_V6', 'Unfiltered_aVF', 'Unfiltered_aVL', 'Unfiltered_aVR']] = unfiltered_half_leads[['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']]

    del unfiltered_half_leads
    del half_VTI_leads

    
if (len(half_data) > 0) and (len(data) > 0):
    df = pd.concat([df, half_df])
    df = df.reset_index(drop=True)
    del half_data
    del data
    del half_df
    
if (len(half_data) > 0) and (len(data) == 0):
    df = half_df
    del half_df
    del half_data
    
if (len(half_data) == 0) and (len(data) > 0):
    df = df
    del data

df['total_xyz'] = ((df.x)**2 + (df.y)**2 + (df.z)**2)**0.5

QRSVTI = []
for n in range(count):  
    QRSVTI.append(trapz(df.total_xyz[int(df.Qonset[1200*n]+(1200*n)):int(df.Qoffset[1200*n]+(1200*n))]))

QRSVTI = np.repeat(QRSVTI, 1200)
df['QRSVTI'] = QRSVTI
del QRSVTI

QRStVTI = []
for n in range(count):  
    QRStVTI.append(trapz(df.total_xyz[int(df.Qonset[1200*n]+(1200*n)):int(df.Toffset[1200*n]+(1200*n))]))

QRStVTI = np.repeat(QRStVTI, 1200)
df['QRStVTI'] = QRStVTI
del QRStVTI

XVTI = []
for n in range(count):  
    XVTI.append(trapz(abs(df.x[int(df.Qonset[1200*n]+(1200*n)):int(df.Qoffset[1200*n]+(1200*n))])))

XVTI = np.repeat(XVTI, 1200)
df['XVTI'] = XVTI
del XVTI

YVTI = []
for n in range(count):  
    YVTI.append(trapz(abs(df.y[int(df.Qonset[1200*n]+(1200*n)):int(df.Qoffset[1200*n]+(1200*n))])))

YVTI = np.repeat(YVTI, 1200)
df['YVTI'] = YVTI
del YVTI

ZVTI = []
for n in range(count):  
    ZVTI.append(trapz(abs(df.z[int(df.Qonset[1200*n]+(1200*n)):int(df.Qoffset[1200*n]+(1200*n))])))

ZVTI = np.repeat(ZVTI, 1200)
df['ZVTI'] = ZVTI
del ZVTI

df['QRS3DArea'] = ((df.XVTI)**2 + (df.YVTI)**2 + (df.ZVTI)**2)**0.5

XtVTI = []
for n in range(count):  
    XtVTI.append(trapz(abs(df.x[int(df.Qonset[1200*n]+(1200*n)):int(df.Toffset[1200*n]+(1200*n))])))

XtVTI = np.repeat(XtVTI, 1200)
df['XtVTI'] = XtVTI
del XtVTI

YtVTI = []
for n in range(count):  
    YtVTI.append(trapz(abs(df.y[int(df.Qonset[1200*n]+(1200*n)):int(df.Toffset[1200*n]+(1200*n))])))

YtVTI = np.repeat(YtVTI, 1200)
df['YtVTI'] = YtVTI
del YtVTI

ZtVTI = []
for n in range(count):  
    ZtVTI.append(trapz(abs(df.z[int(df.Qonset[1200*n]+(1200*n)):int(df.Toffset[1200*n]+(1200*n))])))

ZtVTI = np.repeat(ZtVTI, 1200)
df['ZtVTI'] = ZtVTI
del ZtVTI

df['QRSt3DArea'] = ((df.XtVTI)**2 + (df.YtVTI)**2 + (df.ZtVTI)**2)**0.5


XVTI = []
for n in range(count):  
    XVTI.append(trapz((df.x[int(df.Qonset[1200*n]+(1200*n)):int(df.Qoffset[1200*n]+(1200*n))])))

XVTI = np.repeat(XVTI, 1200)
df['XVector_VTI'] = XVTI
del XVTI

YVTI = []
for n in range(count):  
    YVTI.append(trapz((df.y[int(df.Qonset[1200*n]+(1200*n)):int(df.Qoffset[1200*n]+(1200*n))])))

YVTI = np.repeat(YVTI, 1200)
df['YVector_VTI'] = YVTI
del YVTI

ZVTI = []
for n in range(count):  
    ZVTI.append(trapz((df.z[int(df.Qonset[1200*n]+(1200*n)):int(df.Qoffset[1200*n]+(1200*n))])))

ZVTI = np.repeat(ZVTI, 1200)
df['ZVector_VTI'] = ZVTI
del ZVTI

df['QRS3DVector_Area'] = ((df.XVector_VTI)**2 + (df.YVector_VTI)**2 + (df.ZVector_VTI)**2)**0.5

XtVTI = []
for n in range(count):  
    XtVTI.append(trapz((df.x[int(df.Qonset[1200*n]+(1200*n)):int(df.Toffset[1200*n]+(1200*n))])))

XtVTI = np.repeat(XtVTI, 1200)
df['XtVector_VTI'] = XtVTI
del XtVTI

YtVTI = []
for n in range(count):  
    YtVTI.append(trapz((df.y[int(df.Qonset[1200*n]+(1200*n)):int(df.Toffset[1200*n]+(1200*n))])))

YtVTI = np.repeat(YtVTI, 1200)
df['YtVector_VTI'] = YtVTI
del YtVTI

ZtVTI = []
for n in range(count):  
    ZtVTI.append(trapz((df.z[int(df.Qonset[1200*n]+(1200*n)):int(df.Toffset[1200*n]+(1200*n))])))

ZtVTI = np.repeat(ZtVTI, 1200)
df['ZtVector_VTI'] = ZtVTI
del ZtVTI

df['QRSt3DVector_Area'] = ((df.XtVector_VTI)**2 + (df.YtVector_VTI)**2 + (df.ZtVector_VTI)**2)**0.5

Tamp = []
XTamp = []
YTamp = []
ZTamp = []
TpTe = []
XTpTe = []
YTpTe = []
ZTpTe = []
QpQe = []

for x in range(count):
    if int(df.Tonset[1200*x]+(1200*x)) > int(df.Toffset[1200*x]+(1200*x)):
        XTamp.append(np.nan)
        XTpTe.append(np.nan)
        YTamp.append(np.nan)
        YTpTe.append(np.nan)
        ZTamp.append(np.nan)
        ZTpTe.append(np.nan)
        Tamp.append(np.nan)
        TpTe.append(np.nan)
        Qa = [abs(n) for n in df.total_xyz[int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]].index(max([abs(n) for n in df.total_xyz[int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]]))
        QpQe.append(len([abs(n) for n in df.total_xyz[int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]][Qa:]))
        
    elif df.Tonset[1200*x] == df.Toffset[1200*x]:
        XTamp.append(max([abs(n) for n in df.x[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]]))
        Ta = [abs(n) for n in df.x[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]].index(max([abs(n) for n in df.x[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]]))
        XTpTe.append(len([abs(n) for n in df.x[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]][Ta:]))
        YTamp.append(max([abs(n) for n in df.y[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]]))
        Ta = [abs(n) for n in df.y[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]].index(max([abs(n) for n in df.y[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]]))
        YTpTe.append(len([abs(n) for n in df.y[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]][Ta:]))
        ZTamp.append(max([abs(n) for n in df.z[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]]))
        Ta = [abs(n) for n in df.z[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]].index(max([abs(n) for n in df.z[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]]))
        ZTpTe.append(len([abs(n) for n in df.z[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]][Ta:]))
        Tamp.append(max([abs(n) for n in df.total_xyz[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]]))
        Ta = [abs(n) for n in df.total_xyz[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]].index(max([abs(n) for n in df.total_xyz[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]]))
        TpTe.append(len([abs(n) for n in df.total_xyz[int(df.Tonset[1200*x]+(1200*x))-10:int(df.Toffset[1200*x]+(1200*x))+10]][Ta:]))
        Qa = [abs(n) for n in df.total_xyz[int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]].index(max([abs(n) for n in df.total_xyz[int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]]))
        QpQe.append(len([abs(n) for n in df.total_xyz[int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]][Qa:]))
        
    else:
        XTamp.append(max([abs(n) for n in df.x[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]]))
        Ta = [abs(n) for n in df.x[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]].index(max([abs(n) for n in df.x[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]]))
        XTpTe.append(len([abs(n) for n in df.x[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]][Ta:]))
        YTamp.append(max([abs(n) for n in df.y[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]]))
        Ta = [abs(n) for n in df.y[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]].index(max([abs(n) for n in df.y[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]]))
        YTpTe.append(len([abs(n) for n in df.y[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]][Ta:]))
        ZTamp.append(max([abs(n) for n in df.z[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]]))
        Ta = [abs(n) for n in df.z[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]].index(max([abs(n) for n in df.z[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]]))
        ZTpTe.append(len([abs(n) for n in df.z[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]][Ta:]))
        Tamp.append(max([abs(n) for n in df.total_xyz[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]]))
        Ta = [abs(n) for n in df.total_xyz[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]].index(max([abs(n) for n in df.total_xyz[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]]))
        TpTe.append(len([abs(n) for n in df.total_xyz[int(df.Tonset[1200*x]+(1200*x)):int(df.Toffset[1200*x]+(1200*x))]][Ta:]))
        Qa = [abs(n) for n in df.total_xyz[int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]].index(max([abs(n) for n in df.total_xyz[int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]]))
        QpQe.append(len([abs(n) for n in df.total_xyz[int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]][Qa:]))
    

QpQe = np.repeat(QpQe, 1200)
df['QpQe'] = QpQe

Tamp = np.repeat(Tamp, 1200)
df['Tamp'] = Tamp

XTamp = np.repeat(XTamp, 1200)
df['XTamp'] = XTamp

YTamp = np.repeat(YTamp, 1200)
df['YTamp'] = YTamp

ZTamp = np.repeat(ZTamp, 1200)
df['ZTamp'] = ZTamp

XTpTe = np.repeat(XTpTe, 1200)
df['XTpTe'] = XTpTe

YTpTe = np.repeat(YTpTe, 1200)
df['YTpTe'] = YTpTe

ZTpTe = np.repeat(ZTpTe, 1200)
df['ZTpTe'] = ZTpTe

TpTe = np.repeat(TpTe, 1200)
df['TpTe'] = TpTe

del Tamp
del XTamp
del YTamp
del ZTamp
del XTpTe
del YTpTe
del ZTpTe
del TpTe
del QpQe

temp = df[['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVR', 'aVL', 'aVF', 'x', 'y', 'z', 'total_xyz']]

Qamp = []
for x in range(count):
    
    for i in range(16):

        if min(temp.iloc[:,i][int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]) < 0 and max(temp.iloc[:,i][int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]) > 0:
            Qamp.append(max(temp.iloc[:,i][int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]) - min(temp.iloc[:,i][int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]))

        elif min(temp.iloc[:,i][int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]) > 0 and max(temp.iloc[:,i][int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]) < 0:
            Qamp.append(max(temp.iloc[:,i][int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]) - min(temp.iloc[:,i][int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]))

        elif min(temp.iloc[:,i][int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]) < 0 and max(temp.iloc[:,i][int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]) < 0:
            Qamp.append(min(temp.iloc[:,i][int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]))

        else:
            Qamp.append(max(temp.iloc[:,i][int(df.Qonset[1200*x]+(1200*x)):int(df.Qoffset[1200*x]+(1200*x))]))
            
del temp

XQamp = Qamp[12::16]
XQamp = np.repeat(XQamp, 1200)
YQamp = Qamp[13::16]
YQamp = np.repeat(YQamp, 1200)
ZQamp = Qamp[14::16]
ZQamp = np.repeat(ZQamp, 1200)
Qamp = Qamp[15::16]
Qamp = np.repeat(Qamp, 1200)

df['XQamp'] = XQamp
df['YQamp'] = YQamp
df['ZQamp'] = ZQamp
df['Qamp'] = Qamp

del XQamp
del YQamp
del ZQamp
del Qamp



text_df = df[['ID', 'Name', 'Age', 'Sex', 'Date', 'Time','HeartRate','Pdur','Print','Qrsdur','Qtint','QTC','TpTe','QRSFrontAxis',
       'QRSVTI','XVector_VTI', 'YVector_VTI','ZVector_VTI',
       'QRStVTI', 'XtVTI','YtVTI', 'ZtVTI', 
       'Qamp','XQamp','YQamp', 'ZQamp','Tamp','XTamp', 'YTamp','ZTamp']]

text_df = text_df[::1200]

# text_df.to_csv('Entresto_Final_Data.csv', index=False)
# signal_df.to_pickle('Entresto_Final_ML.pkl')

text_df.to_csv('data.csv', index=False)

for n in range(count):
    
    # pd.DataFrame(text_df.iloc[n,:]).T.to_csv('{}.csv'.format(root_names[n][:-4]), index=False)
    
    x = df.x[n*1200:(n+1)*1200]
    y = df.y[n*1200:(n+1)*1200]
    z = df.z[n*1200:(n+1)*1200]
    rms = df.total_xyz[n*1200:(n+1)*1200]

    fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(15, 8))

    ax.plot(x)
    ax.set_title('Lead X')

    ax1.plot(y)
    ax1.set_title('Lead Y')

    ax2.plot(z)
    ax2.set_title('Lead Z')

    ax3.plot(rms)
    ax3.set_title('XYZ RMS')

    fig.subplots_adjust(hspace=.3)
    fig.subplots_adjust(wspace=.1)

    fig.savefig('{}.png'.format(root_names[n][:-4]), dpi=1800, format='png')

del df