Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

Loading & Understanding the data
df = pd.read_csv(r'C:\ravikora\downloads\onedrive\employee_burnout_analysis')
df.head()
Employee ID	Date of Joining	Gender	Company Type	WFH Setup Available	Designation	Resource Allocation	Mental Fatigue Score	Burn Rate
0	fffe32003000360033003200	2008-09-30	Female	Service	No	2.0	3.0	3.8	0.16
1	fffe3700360033003500	2008-11-30	Male	Service	Yes	1.0	2.0	5.0	0.36
2	fffe31003300320037003900	2008-03-10	Female	Product	Yes	2.0	NaN	5.8	0.49
3	fffe32003400380032003900	2008-11-03	Male	Service	Yes	1.0	1.0	2.6	0.20
4	fffe31003900340031003600	2008-07-24	Female	Service	No	3.0	7.0	6.9	0.52
df.describe()
Designation	Resource Allocation	Mental Fatigue Score	Burn Rate
count	22750.000000	21369.000000	20633.000000	21626.000000
mean	2.178725	4.481398	5.728188	0.452005
std	1.135145	2.047211	1.920839	0.198226
min	0.000000	1.000000	0.000000	0.000000
25%	1.000000	3.000000	4.600000	0.310000
50%	2.000000	4.000000	5.900000	0.450000
75%	3.000000	6.000000	7.100000	0.590000
max	5.000000	10.000000	10.000000	1.000000
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 22750 entries, 0 to 22749
Data columns (total 9 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   Employee ID           22750 non-null  object 
 1   Date of Joining       22750 non-null  object 
 2   Gender                22750 non-null  object 
 3   Company Type          22750 non-null  object 
 4   WFH Setup Available   22750 non-null  object 
 5   Designation           22750 non-null  float64
 6   Resource Allocation   21369 non-null  float64
 7   Mental Fatigue Score  20633 non-null  float64
 8   Burn Rate             21626 non-null  float64
dtypes: float64(4), object(5)
memory usage: 1.6+ MB
df.isnull().sum()
Employee ID                0
Date of Joining            0
Gender                     0
Company Type               0
WFH Setup Available        0
Designation                0
Resource Allocation     1381
Mental Fatigue Score    2117
Burn Rate               1124
dtype: int64
Questions answered
Total number of records = 22750
Total number of columns (features) = 8 (excluding the target feature)
Target Feature = Burn Rate - we will try and predict the burn rate based on the given data
Number of numerical features = 4
Number of categorical features = 5
Missing values = We find missing / null values in Resource Allocation, Mental fatigue scorers and Burn Rate.
Exploratory Data Analysis
Univariate Analysis
Business Questions - to help improve HR processes
Are there any employee IDs repeated or do we have 22750 unique employees data records?
What is the male-female employee distribution in the organization?
What are the company types to which an employee belongs to, and how are they distributed?
For how many employees is WFH available as an option?
From what date is the organization maintaining a record of the employees (OR) which is the employee record with the oldest joining date?
Who / When did the most recent employee join?
What is the distribution of the employees in each of the designation levels? Which designation level has the highest and which has the lowest count of employees?
Do the number of employees hired across the year follow a uniform distribution or do we see a hiring trend in any of the years?
What is minimum/25th percentile/average/75th percentile/maximum/total work hours of all the employees in the organization?
What is minimum/25th percentile/average/75th percentile/maximum mental fatigue score for all the employees in the organization?
What is minimum/25th percentile/average/75th percentile/maximum burnrate for all the employees in the organization?
df['Employee ID'].nunique()
22750
Yes, there are no repeated employee IDs and there are total of 22750 unique employee data records in the given dataset. This avoids the need to remove duplicate employee records.
sns.countplot(data=df,x='Gender')
<AxesSubplot:xlabel='Gender', ylabel='count'>

len(df[df['Gender']=='Male'])/len(df[df['Gender']=='Female'])
0.9104803493449781
df['Gender'].value_counts()
Female    11908
Male      10842
Name: Gender, dtype: int64
There are more female employees compared to Male employees, thoough the difference is very small. The male to female ratio is 0.91, i.e, for every 1 female there is 0.91 male.
sns.countplot(data=df,x='Company Type')
<AxesSubplot:xlabel='Company Type', ylabel='count'>

df['Company Type'].value_counts()
Service    14833
Product     7917
Name: Company Type, dtype: int64
There are 2 company types - Service and Product to which an employee can belong to. Number of employees in Service type is nearly double that of Product type company.
df['WFH Setup Available'].value_counts()
Yes    12290
No     10460
Name: WFH Setup Available, dtype: int64
sns.countplot(data=df,x='WFH Setup Available')
<AxesSubplot:xlabel='WFH Setup Available', ylabel='count'>

WFH is available to more than half the employees in the organization. We will try and determine if WFH option plays an important role in employee burnout in Bivariate Analysis.
Date of Joining is of Object type. We need to first convert it to date type before we can proceed with any operations on the column.

df['Date of Joining'] = pd.to_datetime(df['Date of Joining'])
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 22750 entries, 0 to 22749
Data columns (total 9 columns):
 #   Column                Non-Null Count  Dtype         
---  ------                --------------  -----         
 0   Employee ID           22750 non-null  object        
 1   Date of Joining       22750 non-null  datetime64[ns]
 2   Gender                22750 non-null  object        
 3   Company Type          22750 non-null  object        
 4   WFH Setup Available   22750 non-null  object        
 5   Designation           22750 non-null  float64       
 6   Resource Allocation   21369 non-null  float64       
 7   Mental Fatigue Score  20633 non-null  float64       
 8   Burn Rate             21626 non-null  float64       
dtypes: datetime64[ns](1), float64(4), object(4)
memory usage: 1.6+ MB
As we can see, the Date of Joining is of datetime format.

df[df['Date of Joining']==df['Date of Joining'].min()]
Employee ID	Date of Joining	Gender	Company Type	WFH Setup Available	Designation	Resource Allocation	Mental Fatigue Score	Burn Rate
495	fffe3900340032003800	2008-01-01	Female	Product	Yes	1.0	4.0	6.7	0.50
833	fffe3800300032003200	2008-01-01	Male	Service	No	1.0	2.0	3.4	0.14
1090	fffe31003300310033003200	2008-01-01	Male	Product	No	2.0	5.0	6.4	0.53
1465	fffe3100340033003600	2008-01-01	Male	Service	No	2.0	6.0	5.7	0.52
1620	fffe32003800350038003100	2008-01-01	Female	Product	Yes	0.0	1.0	2.0	0.16
2097	fffe31003100330038003000	2008-01-01	Female	Service	Yes	1.0	3.0	6.2	0.40
2686	fffe32003800310039003800	2008-01-01	Female	Service	Yes	1.0	2.0	3.4	0.27
3310	fffe31003600390035003900	2008-01-01	Female	Service	Yes	2.0	NaN	4.7	0.23
3566	fffe33003300390037003500	2008-01-01	Male	Service	No	3.0	7.0	8.2	0.74
3826	fffe33003000340032003100	2008-01-01	Male	Service	Yes	3.0	6.0	7.1	0.70
5763	fffe31003900310034003900	2008-01-01	Female	Product	Yes	0.0	1.0	0.5	0.02
5886	fffe3500360033003300	2008-01-01	Female	Service	No	4.0	8.0	6.5	0.55
6060	fffe33003200330034003500	2008-01-01	Female	Service	Yes	1.0	2.0	3.5	0.26
6448	fffe32003200370032003600	2008-01-01	Female	Product	Yes	2.0	4.0	3.9	0.23
6560	fffe3800390034003700	2008-01-01	Female	Product	Yes	2.0	3.0	5.1	0.37
6566	fffe31003100350030003100	2008-01-01	Male	Service	No	4.0	NaN	7.2	0.50
6759	fffe33003300340038003900	2008-01-01	Female	Service	No	3.0	5.0	5.2	0.47
7046	fffe33003000380037003100	2008-01-01	Male	Service	Yes	2.0	4.0	5.2	0.33
7599	fffe3500340037003400	2008-01-01	Female	Service	Yes	3.0	4.0	5.9	0.43
8235	fffe32003300330038003000	2008-01-01	Male	Service	No	1.0	4.0	5.6	0.33
8244	fffe3600330030003800	2008-01-01	Female	Service	Yes	0.0	1.0	3.6	0.17
8793	fffe3600330030003400	2008-01-01	Male	Product	Yes	0.0	1.0	3.1	0.24
9659	fffe31003000390031003400	2008-01-01	Female	Service	Yes	2.0	4.0	6.8	0.45
9909	fffe33003400330035003300	2008-01-01	Male	Product	Yes	1.0	3.0	5.9	0.56
10101	fffe3100360034003600	2008-01-01	Female	Service	Yes	1.0	3.0	5.7	0.38
10645	fffe32003400310038003700	2008-01-01	Female	Service	Yes	1.0	3.0	3.2	0.27
11291	fffe32003400390034003100	2008-01-01	Male	Service	No	2.0	5.0	7.3	0.61
11628	fffe33003400360033003600	2008-01-01	Female	Product	Yes	2.0	4.0	5.6	0.45
12302	fffe31003900390031003600	2008-01-01	Female	Product	Yes	0.0	1.0	2.3	0.05
12536	fffe31003800380037003100	2008-01-01	Female	Service	No	1.0	3.0	4.1	0.28
13628	fffe31003100390036003200	2008-01-01	Female	Product	No	0.0	NaN	1.6	0.06
13921	fffe3400300031003900	2008-01-01	Female	Service	Yes	3.0	5.0	NaN	0.50
14497	fffe3400390037003800	2008-01-01	Female	Service	No	4.0	8.0	8.7	0.81
14735	fffe31003000330034003900	2008-01-01	Female	Service	Yes	4.0	7.0	6.9	0.68
14743	fffe32003500310038003000	2008-01-01	Male	Service	No	3.0	4.0	4.1	0.35
14823	fffe33003000360034003300	2008-01-01	Female	Service	No	0.0	2.0	2.5	0.19
15071	fffe31003400390034003700	2008-01-01	Female	Service	No	4.0	7.0	8.6	0.68
15083	fffe31003700350036003700	2008-01-01	Female	Service	Yes	2.0	3.0	6.3	0.40
15191	fffe32003500360030003100	2008-01-01	Female	Service	No	2.0	5.0	5.8	0.48
15520	fffe31003800330033003500	2008-01-01	Male	Service	No	3.0	5.0	6.5	NaN
15774	fffe33003100370038003600	2008-01-01	Male	Product	No	3.0	5.0	6.1	0.51
16444	fffe32003100300035003000	2008-01-01	Female	Product	Yes	0.0	1.0	3.8	0.23
16840	fffe31003600390032003300	2008-01-01	Male	Product	No	4.0	7.0	7.7	0.65
16915	fffe32003700390036003900	2008-01-01	Female	Service	Yes	2.0	4.0	5.0	0.37
17558	fffe33003200380032003000	2008-01-01	Female	Product	No	2.0	3.0	3.7	0.16
18965	fffe32003100390035003300	2008-01-01	Male	Service	Yes	2.0	4.0	3.0	0.18
19912	fffe340030003500	2008-01-01	Female	Service	Yes	2.0	2.0	3.5	NaN
19917	fffe31003800390030003800	2008-01-01	Female	Service	Yes	0.0	2.0	1.7	0.11
19949	fffe31003700310032003300	2008-01-01	Male	Service	No	3.0	7.0	8.7	0.77
20881	fffe32003900340039003600	2008-01-01	Female	Service	Yes	1.0	3.0	4.6	0.27
21805	fffe31003600320031003200	2008-01-01	Female	Service	Yes	2.0	4.0	4.6	0.41
21818	fffe33003400380032003200	2008-01-01	Male	Service	No	3.0	6.0	6.9	0.54
22002	fffe3500340035003600	2008-01-01	Female	Product	Yes	1.0	1.0	1.5	0.10
22219	fffe33003000320036003200	2008-01-01	Female	Service	Yes	1.0	4.0	6.0	0.39
len(df[df['Date of Joining']==df['Date of Joining'].min()])
54
Looks like the earliest Date of Joining record we have are those of 58 employees who joined Jan 1st, 2008.
df[df['Date of Joining']==df['Date of Joining'].max()]
Employee ID	Date of Joining	Gender	Company Type	WFH Setup Available	Designation	Resource Allocation	Mental Fatigue Score	Burn Rate
629	fffe3100380031003700	2008-12-31	Female	Service	No	2.0	4.0	5.7	0.36
794	fffe3200380032003400	2008-12-31	Male	Service	Yes	2.0	4.0	5.0	0.47
1149	fffe32003600390035003600	2008-12-31	Female	Service	Yes	2.0	6.0	7.7	0.76
1686	fffe32003000370031003900	2008-12-31	Male	Service	No	3.0	4.0	6.0	0.49
1706	fffe32003600310039003500	2008-12-31	Male	Product	No	3.0	5.0	8.1	0.67
...	...	...	...	...	...	...	...	...	...
20250	fffe33003100340032003400	2008-12-31	Female	Service	Yes	3.0	5.0	5.3	0.48
20687	fffe3600380035003400	2008-12-31	Female	Service	Yes	2.0	3.0	6.3	0.44
21478	fffe32003300390032003400	2008-12-31	Female	Service	No	1.0	3.0	6.8	0.49
21642	fffe32003600380031003300	2008-12-31	Female	Service	Yes	3.0	6.0	5.8	0.50
22510	fffe33003000340037003900	2008-12-31	Male	Service	No	3.0	6.0	7.3	0.63
61 rows × 9 columns

len(df[df['Date of Joining']==df['Date of Joining'].max()])
61
The most recent record of employees who joined the organization is of December 31st, 2008, of 61 employees in total.
sns.countplot(data=df,x='Designation')
<AxesSubplot:xlabel='Designation', ylabel='count'>

df['Designation'].value_counts()
2.0    7588
3.0    5985
1.0    4881
4.0    2391
0.0    1507
5.0     398
Name: Designation, dtype: int64
Here is a distribution of employees in each of the designation, with Designation level 2 having the maximum employees of 7588 and designation level 5 having the least employee count of 398.
monthly_hires = df['Date of Joining'].dt.month.value_counts().reset_index()
monthly_hires.rename(columns={'Date of Joining':'Count','index':'month'},inplace=True)
monthly_hires
month	Count
0	8	1972
1	10	1970
2	9	1968
3	3	1947
4	7	1911
5	1	1903
6	5	1900
7	4	1861
8	12	1844
9	11	1841
10	2	1832
11	6	1801
monthly_hires.sort_values(by='month',inplace=True)
monthly_hires
month	Count
5	1	1903
10	2	1832
3	3	1947
7	4	1861
6	5	1900
11	6	1801
4	7	1911
0	8	1972
2	9	1968
1	10	1970
9	11	1841
8	12	1844
sns.lmplot(data=monthly_hires,x='month',y='Count')
<seaborn.axisgrid.FacetGrid at 0x7f7bdee8a3d0>

As we can see, the number of hires in different months of 2008 does not really follow a uniform distribution. It ranges between 1800 to 1972.
df.describe()
Designation	Resource Allocation	Mental Fatigue Score	Burn Rate
count	22750.000000	21369.000000	20633.000000	21626.000000
mean	2.178725	4.481398	5.728188	0.452005
std	1.135145	2.047211	1.920839	0.198226
min	0.000000	1.000000	0.000000	0.000000
25%	1.000000	3.000000	4.600000	0.310000
50%	2.000000	4.000000	5.900000	0.450000
75%	3.000000	6.000000	7.100000	0.590000
max	5.000000	10.000000	10.000000	1.000000
The 25th percentile of employee work hours for the company is 3hrs, min = 1hr, mean = 4.48hrs, 75th percentile = 6 hrs and max work hours = 10 hrs.
The 25th percentile of employee mental fatigue score for the company is 4.6, min = 0, mean = 5.72, 75th percentile = 7.1 and max mental fatigue score = 10.
The 25th percentile of employee burnrate for the company is 0.31, min = 0, mean = 0.45, 75th percentile = 0.59 and max burn rate = 1.
Bivariate Analysis
Business Questions - to help improve HR processes
How are the male and female emloyees distributed across the different company types?
Is the WFH option gender specific? ie. is there Gender bias in WFH option made available to employees in the organization?
How is the designation distribution of employees gender-wise? Do we find a pattern indicative of gender bias?
What is the average(median) working hours (resource allocation) of male and female employees?
What is the average(median) mental fatigue levels of male and female employees?
What is the average(median) burnout levels (burnrate) of male and female employees?
Is WFH option available to a particular company type employees or is it avaialable organization-wide?
How is the designation distribution of employees based on thier company types?
What is the average(median) working hours (resource allocation) of the different company types?
What is the average(median) mental fatigue level of the different company types?
What is the average(median) burnout levels (burnrate) of the different company types?
Is the WFH option limited to employees of higher designations or is uniformly distributed across the designation levels?
What is the average(median) working hours (resource allocation) of employees with WFH facilities and those without?
What is the average(median) mental fatigue levels of employees with WFH facilities and those without?
What is the average(median) burnout levels (burnrate) of employees with WFH facilities and those without?
What is the average(median) working hours (resource allocation) of employees in each designation levels?
What is the average(median) mental fatigue level of employees in each designation levels?
What is the average(median) burnout levels (burnrate) of employees in each designation levels?
Is there a positive/negative correlation between the employee work hours (resource allocation) and mental fatigue levels of employees?
Is there a positive/negative correlation between the employee work hours (resource allocation) and burnrate of employees?
Is there a positive/negative correlation between the mental fatigue levels of employees and burnrate of employees?
P.S: We use Median instead of Mean, to negate the effect of outliers.

df.columns
Index(['Employee ID', 'Date of Joining', 'Gender', 'Company Type',
       'WFH Setup Available', 'Designation', 'Resource Allocation',
       'Mental Fatigue Score', 'Burn Rate'],
      dtype='object')
sns.countplot(data=df,x='Company Type',hue='Gender')
<AxesSubplot:xlabel='Company Type', ylabel='count'>

There male-female distribution across the different company types is mostly uniform, indicating that there is no Gender bias.
sns.countplot(data=df,x='WFH Setup Available',hue='Gender')
<AxesSubplot:xlabel='WFH Setup Available', ylabel='count'>

There are more female employees for whom the WFH option is made available in comparison to male employees. WFH seems to have a female-preference.
sns.countplot(data=df,x='Designation',hue='Gender')
<AxesSubplot:xlabel='Designation', ylabel='count'>

Female employees are mainly concentrated in Designations 0,1 and 2 whereas, Male employees dominate the higher designation levels of 3,4 and 5. This indicates lower rates of promotion for female employees in the company.
df.groupby('Gender')['Resource Allocation'].median().plot(kind='bar')
<AxesSubplot:xlabel='Gender'>

The average working hours of male employees is slightly greater than female employees. (5>4)
df.groupby('Gender')['Mental Fatigue Score'].median().plot(kind='bar')
<AxesSubplot:xlabel='Gender'>

Male employees have a higher mental fatigue compared to female employees.
df.groupby('Gender')['Burn Rate'].median().plot(kind='bar')
<AxesSubplot:xlabel='Gender'>

Male employees have a relatively higher burn rate compared to Female employees.
sns.countplot(data=df,x='Company Type',hue='WFH Setup Available')
<AxesSubplot:xlabel='Company Type', ylabel='count'>

In both Service and Product Company type, employees with WFH option available are in proportion.
sns.countplot(data=df,x='Company Type',hue='Designation')
<AxesSubplot:xlabel='Company Type', ylabel='count'>

In both Service and Product company types, max employees are in Designation-2, and lowest headcount is in Designation-5.
df.groupby('Company Type')['Resource Allocation'].median().plot(kind='bar')
<AxesSubplot:xlabel='Company Type'>

Both Product and Service company type have equal average working hours = 4
df.groupby('Company Type')['Mental Fatigue Score'].median().plot(kind='bar')
<AxesSubplot:xlabel='Company Type'>

Both Product and Service company type have equal average mental fatigue score = 5.9
df.groupby('Company Type')['Burn Rate'].median().plot(kind='bar')
<AxesSubplot:xlabel='Company Type'>

Both Product and Service company type have equal average Burn rate = 0.45
sns.countplot(data=df,x='Designation',hue='WFH Setup Available')
<AxesSubplot:xlabel='Designation', ylabel='count'>

WFH Setup is available across the designations. In fact, the lower designations - 0,1 and 2 has a higher WFH option than without WFH in comparison to the higher designation levels 3,4 and 5.
df.groupby('WFH Setup Available')['Resource Allocation'].median().plot(kind='bar')
<AxesSubplot:xlabel='WFH Setup Available'>

The Average working hours of employees without WFH is higher (5 hrs) than employees with WFH option (4 hrs).
df.groupby('WFH Setup Available')['Mental Fatigue Score'].median().plot(kind='bar')
<AxesSubplot:xlabel='WFH Setup Available'>

The median mental fatigue score is higher in employees without a WFH Setup. WFH setup seems to improve employee mental well-being.
df.groupby('WFH Setup Available')['Burn Rate'].median().plot(kind='bar')
<AxesSubplot:xlabel='WFH Setup Available'>

Employees without a WFH setup exhibit higher burn rate compared to employees with a WFH setup. Another indicator of WFH Setup importance in the organization.
df.groupby('Designation')['Resource Allocation'].median().plot(kind='bar')
<AxesSubplot:xlabel='Designation'>

Employees in Designation 5 have the maximum working hours, with over 8+ average hours agaisnt Designation 1, who have the least average working hours.
df.groupby('Designation')['Mental Fatigue Score'].median().plot(kind='bar')
<AxesSubplot:xlabel='Designation'>

Employees in Designation 5 have the maximum mental fatigue, agaisnt Designation 1, who have the least mental fatigue.
df.groupby('Designation')['Burn Rate'].median().plot(kind='bar')
<AxesSubplot:xlabel='Designation'>

Employees in Designation 5 have the maximum burn rate, agaisnt Designation 1, who have the least burn rate.
df.corr()['Mental Fatigue Score'].sort_values(ascending=False).plot(kind='bar')
<AxesSubplot:>

19.There is a strong +ve correlation between Mental fatigue of employees and the employee work hours.

df.corr()['Resource Allocation'].sort_values(ascending=False).plot(kind='bar')
<AxesSubplot:>

20.There is a strong +ve correlation between the number of work hours of employees and their burn rate.

df.corr()['Mental Fatigue Score'].sort_values(ascending=False).plot(kind='bar')
<AxesSubplot:>

21.There is a strong +ve correlation between Mental fatigue of employees and their burn rate.

Multivariate Analysis
Business Questions - to help improve HR processes
What is the correlation between the target feature(burn rate) and rest of the numerical columns? What can we infer from the correlation?
What is the correlation between the different numerical columns? What can we infer from the correlation?
What is the gender-wise distribution of data of employee work hours across different designations?
What is the gender-wise distribution of data of employee mental fatigue across different designations?
What is the gender-wise distribution of data of employee burn rate across different designations?
What is the company type distribution of data of employee work hours across different designations?
What is the company type distribution of data of employee mental fatigue across different designations?
What is the company type distribution of data of employee burn rate across different designations?
What is the WFH Setup availability distribution of data of employee work hours across different designations?
What is the WFH Setup availability distribution of data of employee mental fatigue across different designations?
What is the WFH Setup availability distribution of data of employee burn rate across different designations?
What is the gender-wise distribution of data of employee mental fatigue against the total work hours of employees?
What is the gender-wise distribution of data of employee burn rate agaisnt the total work hours of employees?
What is the Company type distribution of data of employee mental fatigue agaisnt the total work hours of employees?
What is the Company type distribution of data of employee burn rate agaisnt the total work hours of employees?
What is the WFH Setup availability distribution of data of employee mental fatigue agaisnt the total work hours of employees?
What is the WFH Setup availability distribution of data of employee burn rate agaisnt the total work hours of employees?
What is the gender-wise distribution of data of employee mental fatigue against employee burnrate?
What is the Company type distribution of data of employee mental fatigue against employee burnrate?
What is the WFH Setup availability distribution of data of employee mental fatigue against employee burnrate?
df.columns
Index(['Employee ID', 'Date of Joining', 'Gender', 'Company Type',
       'WFH Setup Available', 'Designation', 'Resource Allocation',
       'Mental Fatigue Score', 'Burn Rate'],
      dtype='object')
df.corr()['Burn Rate'].sort_values(ascending=False).plot(kind='bar')
<AxesSubplot:>

There is a strong +ve correlation between burn rate and the rest of the numerical features, in the order: Mental fatigue, Resource allocation and Designation. This implies employees with high mental fatigue exhibit high burn rate. Also, employees who work more hours have higher tendancy to face burnout and finally employees belonging to higher designation levels have higher burnout levels.
sns.heatmap(df.corr(),annot=True,cmap='Oranges')
<AxesSubplot:>

From the correlation between different features we can infer: a) As the employee designation level goes up, so does the number of work hours. b) As the designation level goes up, employee mental fatigue increases, but the correlation is not very strong. c) When employee works longer hours, their mental fatigue increases.
sns.lineplot(data=df,x='Designation',y='Resource Allocation',hue='Gender')
<AxesSubplot:xlabel='Designation', ylabel='Resource Allocation'>

Both male and female employees have worked longer hours as their designation levels go up, with Male employees slightly higher than female employees around 2 and 3 designation levels.
sns.lineplot(data=df,x='Designation',y='Mental Fatigue Score',hue='Gender')
<AxesSubplot:xlabel='Designation', ylabel='Mental Fatigue Score'>

Male employees face higher mental fatigue compapred to female employees, in both male and female employees there is an upward trend of mental fatigue with designation level.
sns.lineplot(data=df,x='Designation',y='Burn Rate',hue='Gender')
<AxesSubplot:xlabel='Designation', ylabel='Burn Rate'>

Male employees face higher burnout compapred to female employees, in both male and female employees there is an upward trend of mental fatigue with designation level.
sns.lineplot(data=df,x='Designation',y='Resource Allocation',hue='Company Type')
<AxesSubplot:xlabel='Designation', ylabel='Resource Allocation'>

There is no difference between male and female employees, both show an upward trend in number of working hours as their designation levels increase.
sns.lineplot(data=df,x='Designation',y='Mental Fatigue Score',hue='Company Type')
<AxesSubplot:xlabel='Designation', ylabel='Mental Fatigue Score'>

There is no difference between male and female employees, both show an upward trend in mental fatigue as their designation levels increase
sns.lineplot(data=df,x='Designation',y='Burn Rate',hue='Company Type')
<AxesSubplot:xlabel='Designation', ylabel='Burn Rate'>

There is no difference between male and female employees, both show an upward trend in burn rate as their designation levels increase
sns.lineplot(data=df,x='Designation',y='Resource Allocation',hue='WFH Setup Available')
<AxesSubplot:xlabel='Designation', ylabel='Resource Allocation'>

Employees with WFH have worked lesser hours compared to those without WFH option, however the number of work hours has an upward trend with designation levels.
sns.lineplot(data=df,x='Designation',y='Mental Fatigue Score',hue='WFH Setup Available')
<AxesSubplot:xlabel='Designation', ylabel='Mental Fatigue Score'>

Employees with WFH have lesser mental fatigue to those without WFH option, however mental fatigue has an upward trend with designation levels.
sns.lineplot(data=df,x='Designation',y='Burn Rate',hue='WFH Setup Available')
<AxesSubplot:xlabel='Designation', ylabel='Burn Rate'>

Employees with WFH have lesser burnout to those without WFH option, however burnout has an upward trend with designation levels.
sns.lineplot(data=df,x='Resource Allocation',y='Mental Fatigue Score',hue='Gender')
<AxesSubplot:xlabel='Resource Allocation', ylabel='Mental Fatigue Score'>

Irrespective of gender, as employees work longer hours, their mental fatigue increases. Both male and female have a similar trend.
sns.lineplot(data=df,x='Resource Allocation',y='Burn Rate',hue='Gender')
<AxesSubplot:xlabel='Resource Allocation', ylabel='Burn Rate'>

Irrespective of gender, as employees work longer hours, their burnout increases. Both male and female have a similar trend.
sns.lineplot(data=df,x='Resource Allocation',y='Mental Fatigue Score',hue='Company Type')
<AxesSubplot:xlabel='Resource Allocation', ylabel='Mental Fatigue Score'>

Irrespective of company type, as employees work longer hours, their mental fatigue increases. Both male and female have a similar trend.
sns.lineplot(data=df,x='Resource Allocation',y='Burn Rate',hue='Company Type')
<AxesSubplot:xlabel='Resource Allocation', ylabel='Burn Rate'>

Irrespective of Company Type, as employees work longer hours, their burn out increases. Both male and female have a similar trend.
sns.lineplot(data=df,x='Resource Allocation',y='Mental Fatigue Score',hue='WFH Setup Available')
<AxesSubplot:xlabel='Resource Allocation', ylabel='Mental Fatigue Score'>

Employees without WFH setup exhibit higher mental fatigue with increasing working hours compared to those with WFH setup.
sns.lineplot(data=df,x='Resource Allocation',y='Burn Rate',hue='WFH Setup Available')
<AxesSubplot:xlabel='Resource Allocation', ylabel='Burn Rate'>

Employees without WFH setup exhibit higher burnouts with increasing working hours compared to those with WFH setup.
sns.lineplot(data=df,x='Mental Fatigue Score',y='Burn Rate',hue='Gender')
<AxesSubplot:xlabel='Mental Fatigue Score', ylabel='Burn Rate'>

Irrespective of Gender, as employees mental fatigue increases, their burn out increases. Both male and female have a similar trend.
sns.lineplot(data=df,x='Mental Fatigue Score',y='Burn Rate',hue='Company Type')
<AxesSubplot:xlabel='Mental Fatigue Score', ylabel='Burn Rate'>

Irrespective of Company Type, as employees mental fatigue increases, their burn out increases. Both male and female have a similar trend.
sns.lineplot(data=df,x='Mental Fatigue Score',y='Burn Rate',hue='WFH Setup Available')
<AxesSubplot:xlabel='Mental Fatigue Score', ylabel='Burn Rate'>

Irrespective of WFH Setup availability, as employees mental fatigue increases, their burn out increases. Both male and female have a similar trend.
Checking Missing Values
df.isnull().sum()
Employee ID                0
Date of Joining            0
Gender                     0
Company Type               0
WFH Setup Available        0
Designation                0
Resource Allocation     1381
Mental Fatigue Score    2117
Burn Rate               1124
dtype: int64
There are null values in the following features,

Resource Allocation - Mode to fill null values
Mental Fatigue Score - Median to fill null values
Burn Rate - Remove records
Detect Outliers
sns.boxplot(data=df,x='Designation')
<AxesSubplot:xlabel='Designation'>

sns.boxplot(data=df,x='Resource Allocation')
<AxesSubplot:xlabel='Resource Allocation'>

sns.boxplot(data=df,x='Mental Fatigue Score')
<AxesSubplot:xlabel='Mental Fatigue Score'>

sns.boxplot(data=df,x='Burn Rate')
<AxesSubplot:xlabel='Burn Rate'>

There are outliers in Mental Fatigue Scores. However, note that the outliers are present before we have treated the null values. We will treat the null values and then check for outliers if any and treat them accordingly.

Feature Engineering
Imputation - handling Null values
1.We will get rid of all the records with null values for burn rate as it is our target feature.

df['Resource Allocation'].mode()
0    4.0
dtype: float64
df['Resource Allocation'].fillna(4,inplace=True)
df['Mental Fatigue Score'].fillna(df['Mental Fatigue Score'].median(),inplace=True)
df.dropna(inplace=True)
df.isnull().sum()
Employee ID             0
Date of Joining         0
Gender                  0
Company Type            0
WFH Setup Available     0
Designation             0
Resource Allocation     0
Mental Fatigue Score    0
Burn Rate               0
dtype: int64
Handling Outliers
sns.boxplot(data=df,x='Designation')
<AxesSubplot:xlabel='Designation'>

sns.boxplot(data=df,x='Resource Allocation')
<AxesSubplot:xlabel='Resource Allocation'>

sns.boxplot(data=df,x='Mental Fatigue Score')
<AxesSubplot:xlabel='Mental Fatigue Score'>

sns.boxplot(data=df,x='Burn Rate')
<AxesSubplot:xlabel='Burn Rate'>

Looks like its still Mental fatigue scores with outliers. Let's check which are the outliers.

def return_outlier(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3-Q1
    lower_fence = Q1-IQR*1.5
    upper_fence = Q3+IQR*1.5
    outliers = []
    for i in df:
        if i<lower_fence or i>upper_fence:
            outliers.append(i)
    return outliers
outlier_df = pd.DataFrame(return_outlier(df['Mental Fatigue Score']))
There are a total of 482 outliers, however we shall proceed without handling it. Note: I have tried handling the outliers but it affects the overall dataset as handling the 482 outliers creates more in terms of the extreme values. These values can be deleted and tested for better results.

Feauture Extraction
df.head()
Employee ID	Date of Joining	Gender	Company Type	WFH Setup Available	Designation	Resource Allocation	Mental Fatigue Score	Burn Rate
0	fffe32003000360033003200	2008-09-30	Female	Service	No	2.0	3.0	3.8	0.16
1	fffe3700360033003500	2008-11-30	Male	Service	Yes	1.0	2.0	5.0	0.36
2	fffe31003300320037003900	2008-03-10	Female	Product	Yes	2.0	4.0	5.8	0.49
3	fffe32003400380032003900	2008-11-03	Male	Service	Yes	1.0	1.0	2.6	0.20
4	fffe31003900340031003600	2008-07-24	Female	Service	No	3.0	7.0	6.9	0.52
We will remove DOJ as the dataset contains only 2008 joinee records. Emp ID is irrelevant to the model. Niether of these features will be used while training our model.

Encoding
We will encode the following features,

Gender
Company Type
WFH Setup Availability
df = pd.get_dummies(data=df,columns=['Gender','Company Type','WFH Setup Available'],drop_first=True)
df.head()
Employee ID	Date of Joining	Designation	Resource Allocation	Mental Fatigue Score	Burn Rate	Gender_Male	Company Type_Service	WFH Setup Available_Yes
0	fffe32003000360033003200	2008-09-30	2.0	3.0	3.8	0.16	0	1	0
1	fffe3700360033003500	2008-11-30	1.0	2.0	5.0	0.36	1	1	1
2	fffe31003300320037003900	2008-03-10	2.0	4.0	5.8	0.49	0	0	1
3	fffe32003400380032003900	2008-11-03	1.0	1.0	2.6	0.20	1	1	1
4	fffe31003900340031003600	2008-07-24	3.0	7.0	6.9	0.52	0	1	0
Modelling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = df.drop(columns=['Employee ID','Date of Joining','Burn Rate'])
y = df['Burn Rate']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Modelling & Hyperparameter Tuning
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from math import sqrt
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")
result = pd.DataFrame(columns=['Model','Mean Absolute Error','Mean Squared Error','Root Mean Squared Error','R2 Score'])
Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_y_pred = lr.predict(X_test)
print('Linear Regression')
print('\n Mean Absolute Error = ')
mae = mean_absolute_error(y_test,lr_y_pred)
print(mae)
print('\n Mean Squared Error')
mse = mean_squared_error(y_test,lr_y_pred)
print(mse)
print('\n Root Mean Squared Error')
rmse = sqrt(mse)
print(rmse)
print('\n R Square = ')
r2 = r2_score(y_test,lr_y_pred)
print(r2)
result.loc[0] = ['Linear Regression',mae,mse,rmse,r2]
Linear Regression

 Mean Absolute Error = 
0.05393232854942025

 Mean Squared Error
0.005112616768299304

 Root Mean Squared Error
0.07150256476728163

 R Square = 
0.8722235249218219
Decision Tree
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
criterion = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
splitter = ['best','random']
max_depth = [None,1,3,5,7,9,11,13,15]
min_samples_leaf = list(range(1,20,1))
max_features = ['auto','log2','sqrt',None]
max_leaf_nodes = [None,10,20,30,40,50,60,70,80,90,100]
min_samples_split = list(range(1,40))
dtr_hyperparameters = dict(criterion=criterion,splitter=splitter,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_features=max_features,max_leaf_nodes=max_leaf_nodes,min_samples_split=min_samples_split)
dtr_gridSearch = RandomizedSearchCV(dtr,param_distributions=dtr_hyperparameters,cv=2,verbose=1)
dtr_gridSearch.fit(X_train,y_train)
dtr_y_pred = dtr_gridSearch.predict(X_test)
print('Decision Tree Regressor')
print('\n Mean Absolute Error = ')
mae = mean_absolute_error(y_test,dtr_y_pred)
print(mae)
print('\n Mean Squared Error')
mse = mean_squared_error(y_test,dtr_y_pred)
print(mse)
print('\n Root Mean Squared Error')
rmse = sqrt(mse)
print(rmse)
print('\n R Square = ')
r2 = r2_score(y_test,dtr_y_pred)
print(r2)
result.loc[1] = ['Decision Tree',mae,mse,rmse,r2]
Fitting 2 folds for each of 10 candidates, totalling 20 fits
Decision Tree Regressor

 Mean Absolute Error = 
0.049692509247842166

 Mean Squared Error
0.004347984741060419

 Root Mean Squared Error
0.06593925038291244

 R Square = 
0.8913335012021224
Support Vector Machines - SVR
from sklearn.svm import SVR
svr = SVR()
C = [0.1,1,10,100,1000,10000] 
degree = [0, 1, 2, 3, 4, 5, 6]
gamma = [1.0,0.5,0.1,0.01,0.001,0.0001,1e-5,1e-6]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
svr_hyperparameters = dict(C=C,degree=degree,gamma=gamma,kernel=kernel)
svr_gridSearch = RandomizedSearchCV(svr,param_distributions=svr_hyperparameters,cv=3,verbose=1)
svr_gridSearch.fit(X_train,y_train)
svr_y_pred = svr_gridSearch.predict(X_test)
print('Decision Tree Regressor')
print('\n Mean Absolute Error = ')
mae = mean_absolute_error(y_test,svr_y_pred)
print(mae)
print('\n Mean Squared Error')
mse = mean_squared_error(y_test,svr_y_pred)
print(mse)
print('\n Root Mean Squared Error')
rmse = sqrt(mse)
print(rmse)
print('\n R Square = ')
r2 = r2_score(y_test,svr_y_pred)
print(r2)
result.loc[2] = ['Support Vector Machines - SVR',mae,mse,rmse,r2]
Fitting 3 folds for each of 10 candidates, totalling 30 fits
Decision Tree Regressor

 Mean Absolute Error = 
0.05319673182493613

 Mean Squared Error
0.0043854214990807746

 Root Mean Squared Error
0.06622251504647626

 R Square = 
0.890397867417119
Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
bootstrap = [True,False]
criterion = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
max_depth = [None,1,3,5,7,9,11,13,15]
min_samples_leaf = list(range(1,20,1))
max_features = ['auto','log2','sqrt',None]
max_leaf_nodes = [None,10,20,30,40,50,60,70,80,90,100]
min_samples_split = list(range(1,40))
n_estimators = list(range(100,2000,100))
warm_start = [True,False]
rfr_hyperparameters = dict(bootstrap=bootstrap,criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,max_features=max_features,max_leaf_nodes=max_leaf_nodes,min_samples_split=min_samples_split,n_estimators=n_estimators,warm_start=warm_start)
rfr_gridSearch = RandomizedSearchCV(rfr,param_distributions=rfr_hyperparameters,cv=3,verbose=1)
rfr_gridSearch.fit(X_train,y_train)
rfr_y_pred = rfr_gridSearch.predict(X_test)
print('Random Forest Regressor')
print('\n Mean Absolute Error = ')
mae = mean_absolute_error(y_test,rfr_y_pred)
print(mae)
print('\n Mean Squared Error')
mse = mean_squared_error(y_test,rfr_y_pred)
print(mse)
print('\n Root Mean Squared Error')
rmse = sqrt(mse)
print(rmse)
print('\n R Square = ')
r2 = r2_score(y_test,rfr_y_pred)
print(r2)
result.loc[3] = ['Random Forest Regressor',mae,mse,rmse,r2]
Fitting 3 folds for each of 10 candidates, totalling 30 fits
Random Forest Regressor

 Mean Absolute Error = 
0.049256335159690974

 Mean Squared Error
0.003991867695191005

 Root Mean Squared Error
0.06318122897816253

 R Square = 
0.9002337147128613
Ridge Regression
from sklearn.linear_model import Ridge
ridge = Ridge()
alpha = [int(x) for x in np.linspace(0.01,0.9,25)] 
ridge_hyperparameters = dict(alpha=alpha)
ridge_gridSearch = RandomizedSearchCV(ridge,param_distributions=ridge_hyperparameters,cv=3,verbose=1)
ridge_gridSearch.fit(X_train,y_train)
ridge_y_pred = ridge_gridSearch.predict(X_test)
print('Ridge Regression')
print('\n Mean Absolute Error = ')
mae = mean_absolute_error(y_test,ridge_y_pred)
print(mae)
print('\n Mean Squared Error')
mse = mean_squared_error(y_test,ridge_y_pred)
print(mse)
print('\n Root Mean Squared Error')
rmse = sqrt(mse)
print(rmse)
print('\n R Square = ')
r2 = r2_score(y_test,ridge_y_pred)
print(r2)
result.loc[4] = ['Ridge Regression',mae,mse,rmse,r2]
Fitting 3 folds for each of 10 candidates, totalling 30 fits
Ridge Regression

 Mean Absolute Error = 
0.05393232854942027

 Mean Squared Error
0.005112616768299308

 Root Mean Squared Error
0.07150256476728166

 R Square = 
0.8722235249218218
Lasso Regression
from sklearn.linear_model import Lasso
lasso = Lasso()
alpha = [int(x) for x in np.linspace(0.01,0.9,25)]
lasso_hyperparameters = dict(alpha=alpha)
lasso_gridSearch = RandomizedSearchCV(lasso,param_distributions=lasso_hyperparameters,cv=3,verbose=1)
lasso_gridSearch.fit(X_train,y_train)
lasso_y_pred = lasso_gridSearch.predict(X_test)
print('Lasso Regression')
print('\n Mean Absolute Error = ')
mae = mean_absolute_error(y_test,lasso_y_pred)
print(mae)
print('\n Mean Squared Error')
mse = mean_squared_error(y_test,lasso_y_pred)
print(mse)
print('\n Root Mean Squared Error')
rmse = sqrt(mse)
print(rmse)
print('\n R Square = ')
r2 = r2_score(y_test,lasso_y_pred)
print(r2)
result.loc[5] = ['Lasso Regression',mae,mse,rmse,r2]
Fitting 3 folds for each of 10 candidates, totalling 30 fits
Lasso Regression

 Mean Absolute Error = 
0.053932328549420254

 Mean Squared Error
0.005112616768299305

 Root Mean Squared Error
0.07150256476728163

 R Square = 
0.8722235249218219
Gradient Boost Regressor
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor()
alpha = np.linspace(0,0.9,9)
criterion = ['friedman_mse', 'squared_error', 'mse', 'mae']
loss = ['squared_error', 'absolute_error', 'huber', 'quantile']
max_features = ['auto', 'sqrt', 'log2']
max_leaf_nodes = [None,1,3,5,7,9,11,13,15]
learning_rate=[1, 0.5, 0.25, 0.1, 0.05, 0.01]
n_estimators=[1, 2, 4, 8, 16, 32, 64, 128, 256,512]
min_samples_split=list(range(1,10,1))
min_samples_leaf=list(range(1,10,1))
max_depth=list(range(1,32,1))
GBR_hyperparameters = dict(alpha=alpha, criterion=criterion, loss=loss, max_features=max_features, max_leaf_nodes=max_leaf_nodes, learning_rate=learning_rate, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, max_depth=max_depth)
GBR_gridSearch = RandomizedSearchCV(GBR,param_distributions=GBR_hyperparameters,cv=3,verbose=1)
GBR_gridSearch.fit(X_train,y_train)
GBR_y_pred = GBR_gridSearch.predict(X_test)
print('Gradient Boost Regressor')
print('\n Mean Absolute Error = ')
mae = mean_absolute_error(y_test,GBR_y_pred)
print(mae)
print('\n Mean Squared Error')
mse = mean_squared_error(y_test,GBR_y_pred)
print(mse)
print('\n Root Mean Squared Error')
rmse = sqrt(mse)
print(rmse)
print('\n R Square = ')
r2 = r2_score(y_test,GBR_y_pred)
print(r2)
result.loc[6] = ['Gradient Boost Regressor',mae,mse,rmse,r2]
Fitting 3 folds for each of 10 candidates, totalling 30 fits
Gradient Boost Regressor

 Mean Absolute Error = 
0.04883136212446744

 Mean Squared Error
0.003997180178785538

 Root Mean Squared Error
0.06322325662907233

 R Square = 
0.9001009430895648
XGBoost Regressor
from xgboost import XGBRegressor
XGBR = XGBRegressor()
max_depth = [3,6,9,12,15]
learning_rate = [int(x) for x in np.linspace(0.01,0.9,25)]
n_estimators = list(range(100,1000,100))
reg_alpha = [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,204.8]
reg_lambda = [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,204.8]
gamma = [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,204.8]
booster =  ['gbtree', 'gblinear']
XGBR_hyperparameters = dict(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, reg_alpha=reg_alpha, reg_lambda=reg_lambda, gamma=gamma, booster=booster)
XGBR_gridSearch = RandomizedSearchCV(XGBR,param_distributions=XGBR_hyperparameters,cv=3,verbose=1)
XGBR_gridSearch.fit(X_train,y_train)
XGBR_y_pred = XGBR_gridSearch.predict(X_test)
print('Extreme Gradient Boosting Regressor (XGBR)')
print('\n Mean Absolute Error = ')
mae = mean_absolute_error(y_test,XGBR_y_pred)
print(mae)
print('\n Mean Squared Error')
mse = mean_squared_error(y_test,XGBR_y_pred)
print(mse)
print('\n Root Mean Squared Error')
rmse = sqrt(mse)
print(rmse)
print('\n R Square = ')
r2 = r2_score(y_test,XGBR_y_pred)
print(r2)
result.loc[7] = ['XGBoost Regressor',mae,mse,rmse,r2]
Fitting 3 folds for each of 10 candidates, totalling 30 fits
[13:48:51] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:48:51] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:48:52] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:48:52] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:48:52] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:48:52] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:49:07] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:49:07] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:49:07] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:49:07] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:49:08] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:49:08] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:49:29] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:49:29] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:49:30] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:49:30] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:49:30] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:49:30] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[13:49:38] WARNING: ../src/learner.cc:576: 
Parameters: { "gamma", "max_depth" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


Extreme Gradient Boosting Regressor (XGBR)

 Mean Absolute Error = 
0.1658908754623921

 Mean Squared Error
0.0425496300863132

 Root Mean Squared Error
0.20627561680022483

 R Square = 
-0.06341664057837137
AdaBoost Regressor
from sklearn.ensemble import AdaBoostRegressor
adaR = AdaBoostRegressor()
n_estimators = list(range(100,2000,100))
learning_rate = [float(x) for x in np.linspace(0.01,0.9,25)]
loss = ['linear', 'square', 'exponential']
AdaR_hyperparameters = dict(learning_rate=learning_rate, n_estimators=n_estimators, loss=loss)
AdaR_gridSearch = RandomizedSearchCV(adaR,param_distributions=AdaR_hyperparameters,cv=3,verbose=1)
AdaR_gridSearch.fit(X_train,y_train)
AdaR_y_pred = AdaR_gridSearch.predict(X_test)
print('AdaBoost Regressor')
print('\n Mean Absolute Error = ')
mae = mean_absolute_error(y_test,AdaR_y_pred)
print(mae)
print('\n Mean Squared Error')
mse = mean_squared_error(y_test,AdaR_y_pred)
print(mse)
print('\n Root Mean Squared Error')
rmse = sqrt(mse)
print(rmse)
print('\n R Square = ')
r2 = r2_score(y_test,AdaR_y_pred)
print(r2)
result.loc[8] = ['AdaBoost Regressor',mae,mse,rmse,r2]
Fitting 3 folds for each of 10 candidates, totalling 30 fits
AdaBoost Regressor

 Mean Absolute Error = 
0.06312079129650573

 Mean Squared Error
0.006326874600658698

 Root Mean Squared Error
0.07954165324318258

 R Square = 
0.8418763284300019
Neural Networks
!pip install keras_tuner
from tensorflow import keras
from keras_tuner import RandomSearch
from keras.layers import Dense,Dropout
from keras.models import Sequential

def build_model(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers',2,20)):
        model.add(Dense(units=hp.Int('units_'+str(i),min_value=32,max_value=512,step=32),activation='relu'))
        model.add(Dense(units=1,activation='linear'))
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',[1e-2,1e-3,1e-4])),loss='mae',metrics=['mean_absolute_error'])
    return model
tuner = RandomSearch(build_model,objective='val_mean_absolute_error',max_trials=5,executions_per_trial=3,directory='my_dir',project_name='ANN Regressor tuning')

tuner.search(X_train,y_train,epochs=100,validation_data=(X_test,y_test))
Trial 5 Complete [00h 12m 57s]
val_mean_absolute_error: 0.12407869597276051

Best val_mean_absolute_error So Far: 0.04977219303448995
Total elapsed time: 01h 02m 24s
result
Model	Mean Absolute Error	Mean Squared Error	Root Mean Squared Error	R2 Score
0	Linear Regression	0.053932	0.005113	0.071503	0.872224
1	Decision Tree	0.049693	0.004348	0.065939	0.891334
2	Support Vector Machines - SVR	0.053197	0.004385	0.066223	0.890398
3	Random Forest Regressor	0.049256	0.003992	0.063181	0.900234
4	Ridge Regression	0.053932	0.005113	0.071503	0.872224
5	Lasso Regression	0.053932	0.005113	0.071503	0.872224
6	Gradient Boost Regressor	0.048831	0.003997	0.063223	0.900101
7	XGBoost Regressor	0.165891	0.042550	0.206276	-0.063417
8	AdaBoost Regressor	0.063121	0.006327	0.079542	0.841876
