import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')
# Specify label column name here
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
cat_cols = modified_data.select_dtypes(exclude=['int', 'float']).columns
cat_cols = set(cat_cols) - {label}

useless_cols = []
for cat_column_features in cat_cols:
    num_cat = modified_data[cat_column_features].nunique()
    if num_cat > 10:
        useless_cols.append(cat_column_features)

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
np.savetxt("data_processed.csv", modified_data_array, delimiter=",")
