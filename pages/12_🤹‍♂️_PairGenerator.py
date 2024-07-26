import streamlit as st
from aux.random_people_choice import random_people_choice, get_teams
import streamlit.components.v1 as components

st.header('Генератор команд')

teams = st.multiselect('Названия команд', [
    'pandas', 'matplotlib', 'seaborn', 
    'sklearn', 'CountVectorizer', 'TfIDFVectorizer',
    'LogRegression', 'Ridge', 'LASSO', 'ElasticNet',
    'Poisson', 'Bernoulli', 'Gauss',
    'MSE', 'MAE', 'MAPE',
    'XGBoost', 'LightGBM', 'CatBoost',
    'Dropout', 'Convolution','Linear',
    'GPT', 'BERT', 'LSTM', 'RNN', 'LSTM', 
    'ResNet', 'Inception', 'DenseNet', 'VGG', 'AlexNet', 'MobileNet', 
    'YOLO', 'FasterRCNN', 'Mask RCNN', 
    'SQL', 'PySpark'
])

names = st.radio(
            ' ', 
            [
                # 'Александр, Екатерина, Ида, Илья, Марина, Никита, Оксана, Константин, Ерлан', 
                'Артемий, Наталья, Савр, Сергей, Влад, Николай, Валерия, Сауле',
                'Александр, Любовь, Анастасия, Ольга, Толубай, Дмитрий, Еркебулан, Константин, Сергей Ч., Сергей К.'
                # add here more names as str
            ]
        )



gen_btn = st.button('Generate')
st.markdown('---------')
if names and len(teams) != 0 and gen_btn:
        # st.write(names.split(', '))
        # st.write(teams)
    pairs = get_teams(names.split(', '), teams)
    for team_name, team_participants in pairs.items():
        st.markdown(f'__{team_name}__:  {(", ".join(team_participants))}')

components.html(
    """
    <!-- Yandex.Metrika counter -->
<script type="text/javascript" >
   (function(m,e,t,r,i,k,a){m[i]=m[i]||function(){(m[i].a=m[i].a||[]).push(arguments)};
   m[i].l=1*new Date();
   for (var j = 0; j < document.scripts.length; j++) {if (document.scripts[j].src === r) { return; }}
   k=e.createElement(t),a=e.getElementsByTagName(t)[0],k.async=1,k.src=r,a.parentNode.insertBefore(k,a)})
   (window, document, "script", "https://mc.yandex.ru/metrika/tag.js", "ym");

   ym(92504528, "init", {
        clickmap:true,
        trackLinks:true,
        accurateTrackBounce:true,
        webvisor:true
   });
</script>
<noscript><div><img src="https://mc.yandex.ru/watch/92504528" style="position:absolute; left:-9999px;" alt="" /></div></noscript>
<!-- /Yandex.Metrika counter -->
""")
