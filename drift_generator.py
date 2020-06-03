import os
import sys
import argparse
import numpy as np
import pandas as pd
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        help="data in csv fromat with headers",
                        required=True,
                        type=path_check)
    parser.add_argument("-l",
                        "--label",
                        help="name of label column",
                        required=False,
                        type=str)
    parser.add_argument("-d",
                        "--drift",
                        help="fraction of samples with drift",
                        required=False,
                        type=float)
    parser.add_argument("-s",
                        "--output_size",
                        help="Number of output samples",
                        required=False,
                        type=float)
    parser.add_argument("-o",
                        "--output_location",
                        help="Location of output samples",
                        required=False,
                        type=float)

    args = parser.parse_args()
    return args


def path_check(arg):
    if not os.path.exists(arg):
        print("File does not exist. Please provide a valid path")
    else:
        return arg


def find_histogram_type(data):
    # Check if the values are numerical
    if not pd.api.types.is_numeric_dtype(data.dtype):
        return 'unknown'

    # Number of unique values
    num_unique_values = len(np.unique(data))

    if num_unique_values > 25:
        histogram_type = 'continuous'
    elif num_unique_values > 1:
        histogram_type = 'categorical'
    else:
        histogram_type = 'unknown'

    return histogram_type


def cat_noise_generator(data, num_samples):
    unique_values = np.unique(data)
    noise = []
    # fraction of samples per unique category
    num_repetitions = int(num_samples/len(unique_values))
    for category in unique_values:
        noise.extend([category]*num_repetitions)
    # Check the length and append with the last category
    if len(noise) != num_samples:
        difference = num_samples - len(noise)
        if difference > 0:
            noise.extend([category]*difference)
        else:
            noise = noise[0:num_samples]
    return noise


def cont_noise_generator(data, num_samples):
    mean_value = np.mean(data)
    min_value = np.min(data)
    max_value = np.min(data)

    # Number of sin waves in the entire length of the data
    num_sin_waves = 5

    # Randomly choose to introduce positive or
    # negative scale noise
    if random.randint(0, 1):
        start_noise_value = (max_value - mean_value)/2
    else:
        start_noise_value = -(mean_value - min_value)/2

    end_noise_value = 2*start_noise_value
    increments = np.abs((end_noise_value - start_noise_value)/num_samples)
    sin_amplitude = (start_noise_value - end_noise_value)/num_sin_waves

    if start_noise_value < end_noise_value:
        const_noise = np.arange(start_noise_value,
                                end_noise_value,
                                increments)
    else:
        const_noise = np.arange(end_noise_value,
                                start_noise_value,
                                increments)
    const_noise = const_noise[0:num_samples]
    # frequency of the sinwave should be num_samples/num_sin_waves
    freq = (2*np.pi)/(num_samples/num_sin_waves)
    sin_noise = np.sin(np.arange(num_samples)*freq)*sin_amplitude
    noise = const_noise + sin_noise
    return noise


def generate_custom_noise(drift_data):
    features = drift_data.columns
    num_samples = len(drift_data)
    # Determine column types and create histograms
    column_types = {}
    noise = {}
    for feature in features:
        column_types[feature] = find_histogram_type(drift_data[feature])
        if (column_types[feature] == 'continuous'):
            noise[feature] = cat_noise_generator(drift_data[feature],
                                                 num_samples)
        elif (column_types[feature] == 'categorical'):
            noise[feature] = cont_noise_generator(drift_data[feature],
                                                  num_samples)

    # Convert noise to a dataframe
    noise = pd.DataFrame.from_dict(noise)
    return noise, column_types


def generate_drift_data(base_data, drift_fraction, output_size):
    drift_data = base_data.sample(n=output_size, replace=True)
    # Generate indexes where drift is introduced
    random_index = random.sample(range(1, output_size),
                                 int(drift_fraction*output_size))

    noise, column_types = generate_custom_noise(drift_data)
    # Reset index for it to add the values in dataframe peoperly
    drift_data.index = pd.RangeIndex(len(drift_data.index))
    # columns in noise data (columns with unknown formats are not included)
    noise_columns = noise.columns

    if list(noise_columns):
        temp_drift_data = drift_data[noise_columns]
        temp_drift_data.iloc[random_index] = \
            temp_drift_data.iloc[random_index] + \
            noise.iloc[random_index]
        drift_data[noise_columns] = temp_drift_data[noise_columns]
    else:
        print('No drift was added to the dataset. ',
              'None of the columns in the dataset',
              ' were recognized as numeric.')

    return drift_data


def main():
    source = parse_args()
    input_dictionary = vars(source)
    print('input_dictionary', input_dictionary)
    # Read the file provided
    raw_data = pd.read_csv(input_dictionary['input'])

    if not input_dictionary['label']:
        print('No label in the input provided')
        base_data = raw_data
    else:
        base_data = raw_data.drop(columns=input_dictionary['label'])

    if not input_dictionary['drift']:
        input_dictionary['drift'] = 0.5

    if not input_dictionary['output_size']:
        input_dictionary['output_size'] = 2000

    if not input_dictionary['output_location']:
        input_dictionary['output_location'] = \
            input_dictionary['input'][0:-4] + \
            "_drift_data.csv"

    drift_data = generate_drift_data(base_data,
                                     input_dictionary['drift'],
                                     input_dictionary['output_size'])

    # write to file
    drift_data.to_csv(input_dictionary['output_location'], index=False)


if __name__ == "__main__":
    main()
