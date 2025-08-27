from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import ipywidgets as widgets
from IPython.display import display, clear_output

# Define the preprocessing steps
categorical_cols_svm = ['Gender', 'CALC']
numerical_cols_svm = ['Height', 'Weight', 'FCVC']

numerical_transformer_svm = StandardScaler()
categorical_transformer_svm = OneHotEncoder(handle_unknown='ignore')

preprocessor_svm = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_svm, numerical_cols_svm),
        ('cat', categorical_transformer_svm, categorical_cols_svm)
    ],
    remainder='passthrough'
)

# Create the SVM model pipeline
svm_pipeline = Pipeline(steps=[('preprocessor', preprocessor_svm),
                              ('classifier', SVC(random_state=42))])

# Recreate the original training data with selected features
features_for_svm = ['Height', 'Weight', 'Gender', 'CALC', 'FCVC']
X_svm_original = df[features_for_svm]
y_svm_original = df['NObeyesdad']

# Split original data for training the pipeline
X_train_svm_original, X_test_svm_original, y_train_svm_original, y_test_svm_original = train_test_split(
    X_svm_original, y_svm_original, test_size=0.2, random_state=42, stratify=y_svm_original
)

# Train the SVM pipeline on the original training data
svm_pipeline.fit(X_train_svm_original, y_train_svm_original)

# Create interactive widgets for user input
height_input = widgets.FloatText(description='Height (m):')
weight_input = widgets.FloatText(description='Weight (kg):')
gender_input = widgets.Dropdown(options=['Male', 'Female'], description='Gender:')
calc_input = widgets.Dropdown(options=['Sometimes', 'no', 'Frequently'], description='CALC:')
fcvc_input = widgets.FloatText(description='FCVC:')
predict_button = widgets.Button(description='Predict Obesity Level')
output_widget = widgets.Output()

# Display the widgets
display(height_input, weight_input, gender_input, calc_input, fcvc_input, predict_button, output_widget)

# Define the prediction function
def predict_obesity(b):
    with output_widget:
        clear_output() # Clear previous output
        try:
            # Get input values from widgets
            height = height_input.value
            weight = weight_input.value
            gender = gender_input.value
            calc = calc_input.value
            fcvc = fcvc_input.value

            # Create a DataFrame with the new data
            new_data = pd.DataFrame({
                'Height': [height],
                'Weight': [weight],
                'Gender': [gender],
                'CALC': [calc],
                'FCVC': [fcvc]
            })

            # Use the trained pipeline to make predictions on the new data
            predictions = svm_pipeline.predict(new_data)

            print("--- Prediction ---")
            print(f"Predicted Obesity Level: {predictions[0]}")

            # Add a simple interpretation based on typical relationships in obesity data (BMI)
            try:
                bmi = new_data.iloc[0]['Weight'] / (new_data.iloc[0]['Height'] ** 2)
                print(f"BMI: {bmi:.2f}")
                if 'Obesity' in predictions[0]:
                    print("Interpretation: Higher weight relative to height (higher BMI) is a strong indicator of obesity.")
                elif 'Overweight' in predictions[0]:
                     print("Interpretation: Overweight is often associated with BMI above the normal range but below obesity levels.")
                elif 'Normal' in predictions[0]:
                    print("Interpretation: Normal weight is typically within a healthy BMI range.")
                elif 'Insufficient' in predictions[0]:
                     print("Interpretation: Insufficient weight is often associated with BMI below the normal range.")

            except ZeroDivisionError:
                print("BMI could not be calculated due to zero height.")

            # Add advice based on FCVC and CALC
            print("\nAdvice based on lifestyle factors:")
            if fcvc < 2.0:
                print("- Increasing your consumption of vegetables can positively impact your health.")
            elif fcvc >= 2.0:
                print("- Your consumption of vegetables is at a good level.")

            if calc in ['Frequently', 'Sometimes']:
                print("- Frequent or sometimes consumption of alcohol can be associated with weight gain. Consider reducing it.")
            elif calc == 'no':
                 print("- Avoiding alcohol consumption is generally beneficial for weight management.")

            print("\nNote: This interpretation and advice are based on general patterns and the model's training data.")

        except Exception as e:
            print(f"An error occurred: {e}")

# Link the button to the prediction function
predict_button.on_click(predict_obesity)
