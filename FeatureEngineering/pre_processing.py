import pandas as pd
import numpy as np

# Please change the file location as needed
file_location = "./Churn_Modeling.csv"
data = pd.read_csv(file_location)
# Please change the label to match dataset
label = 'Exited'


# Rearrage the dataset columns
cols = data.columns.tolist()
colIdx = data.columns.get_loc(label)
# Do nothing if the label is in the 0th position
# Otherwise, change the order of columns to move label to 0th position
if colIdx != 0:
    cols = cols[colIdx:colIdx+1] + cols[0:colIdx] + cols[colIdx+1:]
# Change the order of data so that label is in the 0th column
modified_data = data[cols]

# Remove the useless columns
# Note that below code is a sample
# Please change the value as you see fit

# Maximum categories allowed in a column
# If a column contains more than 10 categories, it is dropped
MAX_CAT_ALLOWED = 10

cat_cols = modified_data.select_dtypes(exclude=['int', 'float']).columns
cat_cols = set(cat_cols) - {label}

useless_cols = []
for cat_column_features in cat_cols:
    num_cat = modified_data[cat_column_features].nunique()
    if num_cat > MAX_CAT_ALLOWED:
        useless_cols.append(cat_column_features)

# If a column contains only 1 catetgory, it is dropped
for feature_column in modified_data.columns:
    num_cat = modified_data[feature_column].nunique()
    if num_cat <= 1:
        useless_cols.append(feature_column)
modified_data = modified_data.drop(useless_cols, axis=1)

# One hot encode and fill missing values
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
    
# Remove label so that it is not encoded
data_without_label = modified_data.drop([label], axis=1)
# Fills missing values with the median value
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numeric_features = data_without_label.select_dtypes(include=['int64',
                                                    'float64']).columns

categorical_features = data_without_label.select_dtypes(exclude=['int64',
                                                            'float64']).columns

# Create the column transformer
preprocessor_cols = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features),
                  ('cat', categorical_transformer, categorical_features)])
# Create a pipeline with the column transformer, note that
# more things can be added to this pipeline in the future
preprocessor = Pipeline(steps=[('preprocessor', preprocessor_cols)])
preprocessor.fit(data_without_label)
modified_data_without_label = preprocessor.transform(data_without_label)
if (type(modified_data_without_label) is not np.ndarray):
    modified_data_without_label = modified_data_without_label.toarray()

modified_data_array = np.concatenate(
    (np.array(modified_data[label]).reshape(-1, 1),
     modified_data_without_label), axis=1)

# Save the processed file, please change preicison in fmt as needed
np.savetxt("data_processed.csv", modified_data_array, delimiter=",", fmt='%1.3f')

# Split the file into train and test (80% train and 20% test)
from sklearn.model_selection import train_test_split
train, test= train_test_split(modified_data_array, test_size=0.2)

# Save the train file, please change preicison in fmt as needed
np.savetxt("train.csv", train, delimiter=",", fmt='%1.3f')

# Save the test file, please change preicison in fmt as needed
np.savetxt("test.csv", test, delimiter=",", fmt='%1.3f')

