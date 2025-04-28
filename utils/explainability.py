import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def explain_model(df, regressors):
    """Generate feature importance for external regressors."""
    importance_df = pd.DataFrame()

    if len(regressors) == 1:
        # Only 1 feature → use variance based
        feature = regressors[0]
        importance = np.var(df[feature].values)
        importance_df = pd.DataFrame({
            'Feature': [feature],
            'Importance': [importance]
        })
    else:
        # Multiple features → train quick RandomForest for importances
        X = df[regressors]
        y = df['y']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        importance_df = pd.DataFrame({
            'Feature': regressors,
            'Importance': model.feature_importances_
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

    return importance_df


def plot_feature_importance(importance_df, top_n=5):
    """Plot Top N important features."""
    top_features = importance_df.head(top_n)
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f"Top {top_n} Important Features",
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig
