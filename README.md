Ebben a GitHub projektben az MLOps (Machine Learning Operations) területét próbálom kicsit felderíteni. A bemutatott eszközök segítik a gépi tanulási modellek hatékony és megbízható telepítését és üzemeltetését.

Főbb elemek

Modellképzés:

Különböző algoritmusok és kódok találhatók a gépi tanulási modellek képzéséhez.
A projekt bemutatja, hogyan lehet a modelleket hatékonyan és pontosan betanítani különböző adatkészleteken.
Modelltelepítés:

Eszközök és folyamatok a modellek telepítéséhez és üzemeltetéséhez.
Példák arra, hogyan lehet a modelleket különböző környezetekben (pl. felhőben vagy helyi szervereken) futtatni.
Infrastruktúra kódolása:

Az infrastruktúra automatizálása és kezelése kód segítségével.
Példák az infrastruktúra kódolására, beleértve a számítási erőforrások és az adattárolók beállítását.
Részletesebb tartalom

CI/CD folyamatok: A projekt bemutatja, hogyan lehet folyamatos integrációs és szállítási (CI/CD) folyamatokat bevezetni a gépi tanulási modellekhez. Ez magában foglalja a modellképzés, -értékelés, -regisztráció és -telepítés automatizálását.
Adatkezelés: Az adatok verziókezelése és nyomon követése, hogy biztosítsuk a modellek reprodukálhatóságát és auditálhatóságát.
Modellmonitorozás: A telepített modellek teljesítményének folyamatos figyelemmel kísérése és szükség esetén újraképzésük.
Ez a projekt hasznos lehet mindazok számára, akik szeretnék megérteni és alkalmazni az MLOps legjobb gyakorlatait a saját gépi tanulási projektjeikben.

Konténerizálás: A projekt egy docker image-be került összecsomagolásra, ez szintén csak egy példaprojekt megoldás, hiszen így pl. ömlesztett logokat kapunk, illetve valamelyik részfeladat hibája esetén a teljes folyamat leállhat. A további fejlesztési iránynak lehetséges a microservice-kre osztás, és, amennyiben docker környezetnél maradunk, akkor docker-compose használata. A kontémerizációs folyamat is optimalizálható, jelen állapotban jelentős erőforrás szükséges az üzemeltetéshez, ez a konténer indítási parancsában is látható a nagy memória hozzárendelésben.

Az alap ML model a https://www.kaggle.com/datasets/rajattomar132/payment-date-dataset alapján készült, szintén bemutató céllal, a kiinduló jupyter notebook Tetovalo munkája, THX. 

Indítási leírás:
        A könyvárban állva az image build:
        "docker build -t mlopsbeadando ."
        A konténer indítása a következő paranccsal lehetséges:
        "docker run --memory=4g --memory-swap=4g -it --rm -p 9090:1080 -p 5102:5102 -p 8501:8501 -p 8080:8080 -e MLFLOW_TRACKING_URI="file:/app/mlruns" mlopsbeadando"

Használat:   
        A konténer indulása után több végponton elérhető a rendszer:
        
        http://127.0.0.1:9090 API végpontok
        
        http://127.0.0.1:5102 MlFlow UI
        
        http://127.0.0.1:8501 Streamlit UI Evidntly elemzéssel 
        
        http://127.0.0.1:8080 USername: admin Password: admin  Airflow UI

A tanítási végponthoz a /data/payment_database.csv szükséges, a predict végpontnál pedig kitöltésre került egy mintaadat, amivel egyszerűen hívható a predikciós folyamat, de természetesen saját adatokkal is kitölthető.

A Streamlit UI esetén a válassz egy csv fájlt: itt tesztdatként a mappa /data/adat.csv választandó, referencia csv-ként pedig ugyanott a ref.csv fájl
        
A feltöltések után aktív a predikciók készítése és a Data Drift jelentés generálása gomb, a jelentés elkészítése után megjelenik a "Data Drift jelentés letöltése" gomb is.

Airfow UI, a készített DAG neve train_and_compare_model.py a demonstrációs cél miatt egyszerűen a konténer /app/data/payment_database.csv fájlt használja a tanítási folyamathoz, majd az eredményről gmail smtp-t használva értesítést küld

A test_train_inference.py a train és az inference szakaszon végez összehasonlítást ugyanazokkal a bemenőadatokkal. A futása több percet is igénybe vehet, a további fejlesztések során beépíthető akár a dashboard részbe is.

Figyelmeztetés:
A kód jelenlegi állapotában demonstrációs célokat szolgál, az érzékeny adatok kezelését nem tartalmazza. A dags/train_and_compare_model.py fájlban az smtp részt saját adatokkal ki kell tölteni a működéshez!
