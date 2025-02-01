import pandas as pd
kd=pd.read_csv('datasets/spam.csv',encoding='latin-1')
kd=kd.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
# print(kd.head())
rd=pd.read_csv('datasets/super_sms_dataset.csv',encoding='latin-1')
# print(rd.head())
columns = list(kd.columns)
columns[0], columns[1] = columns[1], columns[0] 
df = kd[columns]

print(df.head())

# Save the updated DataFrame back to a CSV file
# df.to_csv('swapped_file.csv', index=False)
df.rename(columns={'v2':'Message', 'v1':'Label'}, inplace=True)
rd.rename(columns={'SMSes': 'Message', 'Labels': 'Label'}, inplace=True)
print(df.head())

rd['Label']=rd['Label'].replace({0.0: 'ham', 1.0: 'spam'}).infer_objects(copy=False)

print(rd.columns)
rd = rd.dropna(subset=['Label'])

print(df.head())
print(rd.head())

cd=pd.concat([df,rd],ignore_index=True)
print(cd.head())
cd.to_csv('datasets/combined_file.csv', index=False)