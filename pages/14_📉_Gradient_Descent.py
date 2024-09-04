import streamlit as st
import numpy as np
import plotly.graph_objects as go
st.set_page_config(layout='wide')

def f(x):
    # return x**2
    return x**4 + 4*x**3 - 4*np.tanh(x) - 2*np.cosh(x)

def grad(x):
    
    gradient = 4*x**3 + 12*x**2 - 4*((1/np.cosh(x))**2) - 2*np.sinh(x)
    print(f'input x: {x}, gradient: {gradient}')
    return gradient

def simple_gd(x_init: float):
    x_path = []
    x_path.append(x_init)
    y_path = []
    y_path.append(f(x_path[-1]))
    # lr = 0.1
    global lr

    for _ in range(n_iters):
        x_new = x_path[-1] - lr * grad(x_path[-1])
        y_path.append(f(x_new))
        x_path.append(x_new)
    return x_path, y_path


def momentum(x_init: float, rho: float):
    x_path = []
    x_path.append(x_init)
    y_path = []
    y_path.append(f(x_path[-1]))
    v = []
    v.append(0)
    # lr = 0.1
    # global lr

    for _ in range(n_iters):
        gradient = grad(x_path[-1])
        v.append(rho * v[-1] + gradient)
        x_new = x_path[-1] - lr * v[-1]
        y_path.append(f(x_new))
        x_path.append(x_new)
    return x_path, y_path

def adagrad(x_init: float):
    x_path = []
    x_path.append(x_init)
    y_path = []
    y_path.append(f(x_path[-1]))
    # global lr
    # lr = 0.1
    g_sum = 0
    for _ in range(n_iters):
        gradient = grad(x_path[-1])
        g_sum += gradient**2
        print(f'G_SUM:{g_sum/1000}')
        x_new = x_path[-1] - lr * x_path[-1] / np.sqrt(g_sum/100000 + .001)
        y_path.append(f(x_new))
        x_path.append(x_new)
    return x_path, y_path

col1, col2 = st.columns([.1, .9])

with col1:
    start_x = st.number_input('start point x', -5., 2.1, 1.2, .05)
    lr = st.number_input('learning_rate', 0.00, 1.1, step=.01)
    momemtum_coef = st.number_input('momentum', 0., 1., .9, .1)
    n_iters = st.number_input('N iterations', 1, 200, 20, 1)

with col2:
    alg_c1, alg_c2, alg_c3 = st.columns(3)

    with alg_c1:
        cb_vanilla = st.checkbox('Vanilla gd', value=True)
    with alg_c2:
        cb_momentum = st.checkbox('momentum gd', value=True)
    with alg_c3:
        cb_adagrad = st.checkbox('adagrad gd', value=True)

    x_true = np.linspace(-5, 2, 20)
    y = f(x_true)

    print('For VANILLA')
    
    print('For MOMENTUM')
    
    print('For ADAGRAD')
    

    fig = go.Figure(
        layout_yaxis_range=[-60,50],
        layout_xaxis_range=[-6, 2.5]
        )
    fig.add_trace(go.Scatter(x=x_true, y=y,
                        mode='lines',
                        name='lines'))
    if cb_vanilla:
        x_gd, y_gd = simple_gd(start_x)
        fig.add_trace(go.Scatter(
                            x=x_gd, y=y_gd,
                            mode='lines+markers',
                            name='vanilla gd')
                    )
    if cb_momentum:
        x_ac, y_ac = momentum(start_x, momemtum_coef)
        fig.add_trace(go.Scatter(
                            x=x_ac, y=y_ac,
                            line=dict(dash='dot'),
                            mode='lines+markers',
                            name='momentum')
        )
    if cb_adagrad:
        x_ad, y_ad = adagrad(start_x)
        fig.add_trace(go.Scatter(
                            x=x_ad, y=y_ad,
                            line=dict(dash='dash', color='firebrick'),
                            mode='lines+markers',
                            name='adagrad')
                    )
    st.plotly_chart(fig, use_container_width=True)

