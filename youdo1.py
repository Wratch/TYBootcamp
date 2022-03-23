import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

def main(verbosity=False):
    cal_housing = fetch_california_housing()
    df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    X = df['MedInc']

    st.dataframe(X)
    st.dataframe(y)

    beta, y_pred = my_reg(X, y)

    st.dataframe(y_pred)

    res_df = pd.DataFrame(dict(X=X, y=y, y_pred=y_pred))

    fig = px.scatter(res_df, x="X", y=["y", 'y_pred'])

    st.plotly_chart(fig, use_container_width=True)

    st.latex(fr"Price = {beta[1]:.4f} \times MedInc + {beta[0]:.4f}")
    st.write(f'You do loss  : {mean_squared_error(y, y_pred)}')


def my_reg(X, y, verbose=False):
    print("-------- start --------")
    # My formula : y = Θ(1-e^-(y-b0-b1*x)^2)
    beta = np.random.random(2)
    alpha = 0.002
    n_max_iter = 10000
    theta = 0.001

    for it in range(n_max_iter):
        y_pred: np.ndarray = beta[0] + beta[1] * X

        g_b0 = (-2 * theta * (y - y_pred) * np.exp(-1 * (y - y_pred) ** 2)).sum()
        g_b1 = (-2 * theta * X * (y - y_pred) * np.exp(-1 * (y - y_pred) ** 2)).sum()
        print(f"({it}) beta: {beta}, gradient: {g_b0} {g_b1}")

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        if np.linalg.norm(beta - beta_prev) < alpha/100:
            print(f"I do early stoping at iteration {it}")
            break
    print(f'You do loss  : {mean_squared_error(y, y_pred)}')
    return beta, y_pred


if __name__ == '__main__':
    main(st.sidebar.checkbox("verbosity"))
