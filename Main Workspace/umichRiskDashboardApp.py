
import io, math, json, time, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title = "PriceOnlyRiskDashboardGoBlue", layout = "wide")

st.markdown('''
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%);
    }
    h1 {
        color: #00274C !important;
        font-family: "Arial Black", sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 4px solid #FFCB05;
        padding-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0, 39, 76, 0.1);
    }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #00274C 0%, #003366 100%);
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #FFCB05;
        box-shadow: 0 6px 12px rgba(0, 39, 76, 0.2);
    }
    [data-testid="metric-container"] label {
        color: #FFCB05 !important;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 12px !important;
        letter-spacing: 1px;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: white !important;
        font-size: 28px !important;
        font-weight: bold;
    }
    [data-testid="stSidebar"] {
        background: #00274C;
    }
    [data-testid="stSidebar"] h2 {
        color: #FFCB05 !important;
    }
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #FFCB05 0%, #FFD733 100%);
        color: #00274C;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #FFD733 0%, #FFCB05 100%);
        box-shadow: 0 4px 8px rgba(255, 203, 5, 0.3);
        transform: translateY(-2px);
    }
    .stAlert {
        background: rgba(0, 39, 76, 0.05);
        border: 1px solid #00274C;
        border-left: 5px solid #FFCB05;
    }
</style>
''', unsafe_allow_html = True)

st.markdown('''
<div style="text-align: center; margin-bottom: 30px;">
    <div style="display: inline-block; padding: 20px 40px; background: #00274C; border-radius: 10px; border: 3px solid #FFCB05;">
        <h2 style="color: #FFCB05; margin: 0; font-family: Arial Black; letter-spacing: 3px;">RISK ANALYTICS DASHBOARD</h2>
        <p style="color: white; margin: 5px 0 0 0; font-size: 14px; letter-spacing: 2px;">MICHIGAN FINANCIAL ANALYTICS</p>
    </div>
</div>
''', unsafe_allow_html = True)


def fetchAlphaVantage(symbol: str, apiKey: str, years: int = 3) -> pd.DataFrame:
    url = "https://www.alphavantage.co/query"
    parameters = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": apiKey
    }
    response = requests.get(url, params = parameters, timeout = 30)
    data = response.json()
    timeSeriesKey = "Time Series (Daily)"

    if timeSeriesKey not in data:
        message = data.get("Note") or data.get("Error Message") or "Unknown error"
        raise RuntimeError(message)

    frame = pd.DataFrame.from_dict(data[timeSeriesKey], orient = "index").apply(pd.to_numeric, errors = "coerce")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()

    renameMap = {
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    }
    frame = frame.rename(columns = renameMap)

    cutoffDate = datetime.now() - timedelta(days = 365 * years + 5)
    frame = frame[frame.index >= cutoffDate]

    keepColumns = ["Open", "High", "Low", "Close", "Volume"]
    existingColumns = [column for column in keepColumns if column in frame.columns]
    frame = frame[existingColumns].dropna(how = "any")

    if frame.empty:
        raise RuntimeError("No recent data returned for " + symbol + " after trimming to " + str(years) + " years.")

    return frame


def computeReturns(priceSeries: pd.Series, useLog: bool = True) -> pd.Series:
    if useLog:
        returnSeries = np.log(priceSeries).diff()
    else:
        returnSeries = priceSeries.pct_change()
    return returnSeries.dropna()


def fitStudentT(returnSeries: pd.Series):
    degreesFreedom, locationValue, scaleValue = stats.t.fit(returnSeries.values)
    return degreesFreedom, locationValue, scaleValue


def buildTPdfOverlay(returnSeries: pd.Series, degreesFreedom: float, locationValue: float, scaleValue: float, binCount: int = 50):
    histogramHeights, histogramEdges = np.histogram(returnSeries, bins = binCount, density = True)
    histogramMidpoints = 0.5 * (histogramEdges[1:] + histogramEdges[:-1])
    evaluationGrid = np.linspace(histogramMidpoints.min(), histogramMidpoints.max(), 400)
    pdfValues = stats.t.pdf((evaluationGrid - locationValue) / scaleValue, degreesFreedom) / scaleValue
    return histogramMidpoints, histogramHeights, evaluationGrid, pdfValues


def computeVarCvar(alphaLevel: float, degreesFreedom: float, locationValue: float, scaleValue: float):
    quantileValue = stats.t.ppf(alphaLevel, degreesFreedom, loc = locationValue, scale = scaleValue)
    evaluationGrid = np.linspace(quantileValue - 10 * scaleValue, quantileValue, 4000)
    pdfValues = stats.t.pdf((evaluationGrid - locationValue) / scaleValue, degreesFreedom) / scaleValue
    expectedShortfall = np.trapz(evaluationGrid * pdfValues, evaluationGrid) / alphaLevel
    return quantileValue, expectedShortfall


def computeSharpeApproximation(returnSeries: pd.Series, tradingDays: int = 252):
    meanDaily = returnSeries.mean()
    deviationDaily = returnSeries.std(ddof = 1)
    meanAnnual = meanDaily * tradingDays
    deviationAnnual = deviationDaily * math.sqrt(tradingDays)

    if deviationAnnual == 0:
        sharpeValue = 0.0
    else:
        sharpeValue = meanAnnual / deviationAnnual

    return meanDaily, deviationDaily, meanAnnual, deviationAnnual, sharpeValue


def computeRollingVolatility(returnSeries: pd.Series, windowSize: int = 30, tradingDays: int = 252) -> pd.Series:
    return returnSeries.rolling(windowSize).std(ddof = 1) * math.sqrt(tradingDays)


def computeDrawdownStats(priceSeries: pd.Series):
    runningPeaks = priceSeries.cummax()
    drawdownSeries = priceSeries / runningPeaks - 1.0
    maximumDrawdown = drawdownSeries.min()
    currentDrawdown = drawdownSeries.iloc[-1]

    isDrawdown = drawdownSeries < 0
    durations = []
    currentDuration = 0

    for isInDrawdown in isDrawdown.values:
        if isInDrawdown:
            currentDuration = currentDuration + 1
        else:
            if currentDuration > 0:
                durations.append(currentDuration)
            currentDuration = 0

    if currentDuration > 0:
        durations.append(currentDuration)

    if durations:
        longestDuration = int(max(durations))
    else:
        longestDuration = 0

    return drawdownSeries, float(maximumDrawdown), float(currentDrawdown), longestDuration


def labelVolatilityRegime(rollingVolatilitySeries: pd.Series, lowQuantile: float = 0.33, highQuantile: float = 0.66):
    cleanedSeries = rollingVolatilitySeries.dropna()
    if cleanedSeries.empty:
        return "NotAvailable", float("nan"), float("nan")

    latestValue = cleanedSeries.iloc[-1]
    lowThreshold = cleanedSeries.quantile(lowQuantile)
    highThreshold = cleanedSeries.quantile(highQuantile)

    if latestValue <= lowThreshold:
        regimeLabel = "LowVol"
    elif latestValue >= highThreshold:
        regimeLabel = "HighVol"
    else:
        regimeLabel = "MediumVol"

    return regimeLabel, float(lowThreshold), float(highThreshold)


def labelTrend(priceSeries: pd.Series, fastWindow: int = 50, slowWindow: int = 200):
    movingAverageFast = priceSeries.rolling(fastWindow).mean()
    movingAverageSlow = priceSeries.rolling(slowWindow).mean()

    if math.isnan(movingAverageFast.iloc[-1]) or math.isnan(movingAverageSlow.iloc[-1]):
        return "NotAvailable", movingAverageFast, movingAverageSlow

    if movingAverageFast.iloc[-1] > movingAverageSlow.iloc[-1]:
        return "UptrendMA" + str(fastWindow) + "Above" + str(slowWindow), movingAverageFast, movingAverageSlow

    if movingAverageFast.iloc[-1] < movingAverageSlow.iloc[-1]:
        return "DowntrendMA" + str(fastWindow) + "Below" + str(slowWindow), movingAverageFast, movingAverageSlow

    return "NeutralTrend", movingAverageFast, movingAverageSlow


def simulateStudentTPaths(startPrice: float, dayCount: int, pathCount: int, degreesFreedom: float, locationValue: float, scaleValue: float, seedValue: int = 42) -> np.ndarray:
    randomGenerator = np.random.default_rng(seedValue)
    returnArray = stats.t.rvs(degreesFreedom, loc = locationValue, scale = scaleValue, size = (dayCount, pathCount), random_state = randomGenerator)
    pathArray = np.empty_like(returnArray)

    pathArray[0, :] = startPrice * np.exp(returnArray[0, :])

    for timeIndex in range(1, dayCount):
        pathArray[timeIndex, :] = pathArray[timeIndex - 1, :] * np.exp(returnArray[timeIndex, :])

    return pathArray


def readSidebarInputs():
    with st.sidebar:
        st.markdown("<h2 style='color: #FFCB05;'>ConfigurationPanel</h2>", unsafe_allow_html = True)

        symbolInput = st.text_input("TickerSymbol", value = "AAPL").strip().upper()
        yearsInput = st.slider("HistoricalYears", 2, 10, 3, 1)
        rollingWindowInput = st.slider("RollingVolatilityWindowDays", 10, 90, 30, 5)
        movingAverageFastWindow = st.slider("FastMovingAverageDays", 10, 100, 50, 5)
        movingAverageSlowWindow = st.slider("SlowMovingAverageDays", 100, 300, 200, 10)
        alphaChoice = st.selectbox("VarCvarTailAlpha", options = [0.01, 0.025, 0.05], index = 2)
        visiblePathCount = st.slider("MonteCarloPathsDisplay", 5, 100, 25, 5)
        histogramPathCount = st.select_slider("MonteCarloSimulations", options = [10000, 20000, 50000, 100000], value = 50000)
        simulationDayCount = st.select_slider("SimulationHorizonDays", options = [126, 189, 252], value = 252)
        seedInput = st.number_input("RandomSeed", value = 42, step = 1)
        apiKeyInput = st.text_input("AlphaVantageApiKey", value = os.getenv("ALPHAVANTAGE_API_KEY", ""), type = "password")
        st.markdown("<br>", unsafe_allow_html = True)
        runButton = st.button("Analyze", use_container_width = True)

    return {
        "symbol": symbolInput,
        "years": yearsInput,
        "rollingWindow": rollingWindowInput,
        "fastWindow": movingAverageFastWindow,
        "slowWindow": movingAverageSlowWindow,
        "alpha": alphaChoice,
        "visiblePaths": visiblePathCount,
        "histogramPaths": histogramPathCount,
        "simulationDays": simulationDayCount,
        "seed": seedInput,
        "apiKeyInput": apiKeyInput,
        "run": runButton
    }


def loadPriceSeries(inputParameters):
    if not inputParameters["run"]:
        st.info("Set parameters in the sidebar and select Analyze.")
        return None

    apiKey = inputParameters["apiKeyInput"] or os.getenv("ALPHAVANTAGE_API_KEY")

    if not apiKey:
        st.error("Alpha Vantage api key is required.")
        return None

    try:
        with st.spinner("Loading " + str(inputParameters["years"]) + " years of data for " + inputParameters["symbol"] + "..."):
            priceFrame = fetchAlphaVantage(inputParameters["symbol"], apiKey, years = inputParameters["years"])
    except Exception as exceptionValue:
        st.error("Error fetching data for " + inputParameters["symbol"] + ": " + str(exceptionValue))
        return None

    priceSeries = priceFrame["Close"].copy()
    priceSeries.index = pd.to_datetime(priceSeries.index)

    if getattr(priceSeries.index, "tz", None) is not None:
        priceSeries = priceSeries.tz_localize(None)

    return priceSeries


def computeAllMetrics(priceSeries, inputParameters):
    returnSeries = computeReturns(priceSeries, useLog = True)

    degreesFreedom, locationValue, scaleValue = fitStudentT(returnSeries)
    meanDaily, deviationDaily, meanAnnual, deviationAnnual, sharpeValue = computeSharpeApproximation(returnSeries, tradingDays = 252)
    varValue, cvarValue = computeVarCvar(inputParameters["alpha"], degreesFreedom, locationValue, scaleValue)

    rollingVolatilitySeries = computeRollingVolatility(returnSeries, windowSize = inputParameters["rollingWindow"], tradingDays = 252)
    volatilityRegimeLabel, lowVolatilityThreshold, highVolatilityThreshold = labelVolatilityRegime(rollingVolatilitySeries)

    drawdownSeries, maximumDrawdown, currentDrawdown, drawdownDays = computeDrawdownStats(priceSeries)
    trendLabelValue, movingAverageFast, movingAverageSlow = labelTrend(priceSeries, fastWindow = inputParameters["fastWindow"], slowWindow = inputParameters["slowWindow"])

    simulationPathsVisible = simulateStudentTPaths(priceSeries.iloc[-1], dayCount = inputParameters["simulationDays"], pathCount = inputParameters["visiblePaths"], degreesFreedom = degreesFreedom, locationValue = locationValue, scaleValue = scaleValue, seedValue = inputParameters["seed"])
    simulationFinalsArray = simulateStudentTPaths(priceSeries.iloc[-1], dayCount = inputParameters["simulationDays"], pathCount = inputParameters["histogramPaths"], degreesFreedom = degreesFreedom, locationValue = locationValue, scaleValue = scaleValue, seedValue = inputParameters["seed"] + 1)[-1, :]

    percentileLow, percentileHigh = np.percentile(simulationFinalsArray, [1.25, 98.75])
    trimmedFinals = simulationFinalsArray[(simulationFinalsArray >= percentileLow) & (simulationFinalsArray <= percentileHigh)]

    histogramMidpoints, histogramHeights, evaluationGrid, pdfValues = buildTPdfOverlay(returnSeries, degreesFreedom, locationValue, scaleValue, binCount = 50)

    return {
        "returnSeries": returnSeries,
        "degreesFreedom": degreesFreedom,
        "locationValue": locationValue,
        "scaleValue": scaleValue,
        "meanDaily": meanDaily,
        "deviationDaily": deviationDaily,
        "meanAnnual": meanAnnual,
        "deviationAnnual": deviationAnnual,
        "sharpeValue": sharpeValue,
        "varValue": varValue,
        "cvarValue": cvarValue,
        "rollingVolatilitySeries": rollingVolatilitySeries,
        "volatilityRegimeLabel": volatilityRegimeLabel,
        "lowVolatilityThreshold": lowVolatilityThreshold,
        "highVolatilityThreshold": highVolatilityThreshold,
        "drawdownSeries": drawdownSeries,
        "maximumDrawdown": maximumDrawdown,
        "currentDrawdown": currentDrawdown,
        "drawdownDays": drawdownDays,
        "trendLabel": trendLabelValue,
        "movingAverageFast": movingAverageFast,
        "movingAverageSlow": movingAverageSlow,
        "simulationPathsVisible": simulationPathsVisible,
        "simulationFinalsArray": simulationFinalsArray,
        "trimmedFinals": trimmedFinals,
        "histogramMidpoints": histogramMidpoints,
        "histogramHeights": histogramHeights,
        "evaluationGrid": evaluationGrid,
        "pdfValues": pdfValues
    }


def renderDashboard(priceSeries, inputParameters, metrics):
    st.title("WolverineRiskAnalyticsSystem")

    metricColumnOne, metricColumnTwo, metricColumnThree, metricColumnFour = st.columns(4)
    metricColumnOne.metric("CurrentPrice", "$" + format(priceSeries.iloc[-1], ",.2f"))
    metricColumnTwo.metric("AnnualReturn", "{:.2f}%".format(100 * metrics["meanAnnual"]))
    metricColumnThree.metric("AnnualVolatility", "{:.2f}%".format(100 * metrics["deviationAnnual"]))
    metricColumnFour.metric("SharpeRatio", "{:.2f}".format(metrics["sharpeValue"]))

    tailConfidence = int((1 - inputParameters["alpha"]) * 100)
    metricColumnFive, metricColumnSix, metricColumnSeven, metricColumnEight = st.columns(4)
    metricColumnFive.metric(str(tailConfidence) + "PercentOneDayVar", "{:.2f}%".format(100 * metrics["varValue"]))
    metricColumnSix.metric(str(tailConfidence) + "PercentOneDayCvar", "{:.2f}%".format(100 * metrics["cvarValue"]))
    metricColumnSeven.metric("Skewness", "{:.2f}".format(stats.skew(metrics["returnSeries"])))
    metricColumnEight.metric("ExcessKurtosis", "{:.2f}".format(stats.kurtosis(metrics["returnSeries"], fisher = True)))

    jarqueStatistic, jarquePValue = stats.jarque_bera(metrics["returnSeries"])
    st.caption("JarqueBeraNormalityTest Statistic " + "{:.2f}".format(jarqueStatistic) + " PValue " + "{:.4f}".format(jarquePValue))

    umichLayout = dict(
        plot_bgcolor = "rgba(245, 245, 245, 0.8)",
        paper_bgcolor = "white",
        font = dict(color = "#00274C"),
        title_font = dict(size = 20, color = "#00274C"),
        showlegend = True,
        hovermode = "x unified",
        xaxis = dict(gridcolor = "rgba(0, 39, 76, 0.1)", showgrid = True),
        yaxis = dict(gridcolor = "rgba(0, 39, 76, 0.1)", showgrid = True)
    )

    priceFigure = go.Figure()
    priceFigure.add_trace(go.Scatter(x = priceSeries.index, y = priceSeries, name = "ClosePrice", mode = "lines", line = dict(color = "#00274C", width = 2)))
    if metrics["movingAverageFast"].notna().sum() > 0:
        priceFigure.add_trace(go.Scatter(x = metrics["movingAverageFast"].index, y = metrics["movingAverageFast"], name = "FastMovingAverage", mode = "lines", line = dict(color = "#FFCB05", width = 2)))
    if metrics["movingAverageSlow"].notna().sum() > 0:
        priceFigure.add_trace(go.Scatter(x = metrics["movingAverageSlow"].index, y = metrics["movingAverageSlow"], name = "SlowMovingAverage", mode = "lines", line = dict(color = "rgba(255, 203, 5, 0.6)", width = 2, dash = "dash")))
    priceFigure.update_layout(title = inputParameters["symbol"] + " PriceAndTrend " + metrics["trendLabel"], xaxis_title = "Date", yaxis_title = "Price", **umichLayout)
    st.plotly_chart(priceFigure, use_container_width = True)

    drawdownFigure = go.Figure()
    drawdownFigure.add_trace(go.Scatter(x = metrics["drawdownSeries"].index, y = 100 * metrics["drawdownSeries"], fill = "tozeroy", name = "DrawdownPercent", mode = "lines", line = dict(color = "#00274C"), fillcolor = "rgba(0, 39, 76, 0.3)"))
    drawdownTitle = inputParameters["symbol"] + " Drawdown Max " + "{:.1f}%".format(100 * metrics["maximumDrawdown"]) + " Current " + "{:.1f}%".format(100 * metrics["currentDrawdown"]) + " Longest " + str(metrics["drawdownDays"]) + " Days"
    drawdownFigure.update_layout(title = drawdownTitle, xaxis_title = "Date", yaxis_title = "DrawdownPercent", **umichLayout)
    st.plotly_chart(drawdownFigure, use_container_width = True)

    volatilityFigure = go.Figure()
    volatilityFigure.add_trace(go.Scatter(x = metrics["rollingVolatilitySeries"].index, y = 100 * metrics["rollingVolatilitySeries"], name = "RollingVolatility", mode = "lines", line = dict(color = "#00274C", width = 2)))
    if not math.isnan(metrics["lowVolatilityThreshold"]):
        volatilityFigure.add_hline(y = 100 * metrics["lowVolatilityThreshold"], line = dict(color = "#FFCB05", width = 2, dash = "dash"), annotation_text = "LowVolThreshold")
    if not math.isnan(metrics["highVolatilityThreshold"]):
        volatilityFigure.add_hline(y = 100 * metrics["highVolatilityThreshold"], line = dict(color = "#FFCB05", width = 2, dash = "dash"), annotation_text = "HighVolThreshold")
    volatilityFigure.update_layout(title = inputParameters["symbol"] + " VolatilityRegime " + metrics["volatilityRegimeLabel"], xaxis_title = "Date", yaxis_title = "AnnualizedVolatilityPercent", **umichLayout)
    st.plotly_chart(volatilityFigure, use_container_width = True)

    distributionFigure = go.Figure()
    distributionFigure.add_trace(go.Bar(x = metrics["histogramMidpoints"], y = metrics["histogramHeights"], name = "ReturnDistribution", marker = dict(color = "#00274C", opacity = 0.7)))
    distributionFigure.add_trace(go.Scatter(x = metrics["evaluationGrid"], y = metrics["pdfValues"], name = "StudentTFit", mode = "lines", line = dict(color = "#FFCB05", width = 3)))
    distributionFigure.update_layout(title = inputParameters["symbol"] + " DailyLogReturns StudentTDegrees " + "{:.1f}".format(metrics["degreesFreedom"]), xaxis_title = "DailyLogReturn", yaxis_title = "Density", **umichLayout)
    st.plotly_chart(distributionFigure, use_container_width = True)

    pathFigure = go.Figure()
    visiblePathCount = metrics["simulationPathsVisible"].shape[1]
    for pathIndex in range(visiblePathCount):
        colorIntensity = pathIndex / visiblePathCount
        lineColor = "rgba(0, 39, 76, " + str(0.3 + 0.5 * colorIntensity) + ")"
        pathFigure.add_trace(go.Scatter(x = np.arange(1, inputParameters["simulationDays"] + 1), y = metrics["simulationPathsVisible"][:, pathIndex], mode = "lines", line = dict(color = lineColor, width = 1), showlegend = False))
    pathFigure.update_layout(title = inputParameters["symbol"] + " MonteCarloSimulation " + str(inputParameters["simulationDays"]) + " TradingDays " + str(visiblePathCount) + " Paths", xaxis_title = "Day", yaxis_title = "SimulatedPrice", **umichLayout)
    st.plotly_chart(pathFigure, use_container_width = True)

    histogramFigure = go.Figure()
    histogramFigure.add_trace(go.Histogram(x = metrics["trimmedFinals"], nbinsx = 30, marker = dict(color = "#00274C", line = dict(color = "#FFCB05", width = 2))))
    histogramTitle = inputParameters["symbol"] + " TerminalPriceDistribution Current " + "{:.2f}".format(priceSeries.iloc[-1]) + " Sims " + "{:,}".format(inputParameters["histogramPaths"])
    histogramFigure.update_layout(title = histogramTitle, xaxis_title = "SimulatedTerminalPrice", yaxis_title = "Frequency", **umichLayout)
    st.plotly_chart(histogramFigure, use_container_width = True)

    st.markdown("""
    <div style='border-top: 3px solid #FFCB05; margin-top: 30px; padding-top: 20px; text-align: center;'>
        <p style='color: #00274C; font-style: italic;'>
        StudentT returns capture fat tails. Var and Cvar are one day return measures. MonteCarlo uses StudentT daily log returns.
        Data Alpha Vantage. Risk free rate assumed near zero.
        </p>
        <p style='color: #FFCB05; font-weight: bold; margin-top: 10px;'>GoBlue</p>
    </div>
    """, unsafe_allow_html = True)


inputParameters = readSidebarInputs()
priceSeries = loadPriceSeries(inputParameters)
if priceSeries is not None:
    metrics = computeAllMetrics(priceSeries, inputParameters)
    renderDashboard(priceSeries, inputParameters, metrics)
