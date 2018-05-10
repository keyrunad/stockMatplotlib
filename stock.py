import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# index 251 is the December 30th, 2016, i.e. last trading day of 2016
# index 250 is the 1st January 2017, i.e first trading day of 2017
# each year has 251 trading days

# close1, open1, high1, volume1 files have data for Twitter
# close, open, high, low, volume files have data for Facebook

# functions which have showplot parameter, if showplot = 'yes', they will display plots, else, they will return decisions only without displaying plots

'''
close = np.loadtxt("close1")
open = np.loadtxt("open1")
high = np.loadtxt("high1")
low = np.loadtxt("low1")
volume = np.loadtxt("volume1")
'''
close = np.loadtxt("close")
open = np.loadtxt("open")
high = np.loadtxt("high")
low = np.loadtxt("low")
volume = np.loadtxt("volume")

days = np.loadtxt("days", dtype=bytes, delimiter="\n").astype(str)

def plotann(decision, x, y):
    # function to annotate buys and sells from decision array
    buyindex = []
    sellindex = []
    for k in range(len(decision)): # get indexes of buys or sells from decision array
        if decision[k] == 'buy':
            buyindex.append(k)
        if decision[k] == 'sell':
            sellindex.append(k)
    for i in buyindex: # annotate 'Buy' on all buy indexes
        plt.annotate('Buy', (x[i], y[i]), xytext=(-20, 40), textcoords='offset points', arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='right', verticalalignment='bottom')
    for i in sellindex: # annotate 'Sell' on all sell indexes
        plt.annotate('Sell', (x[i], y[i]), xytext=(20, -40), textcoords='offset points', arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='left', verticalalignment='bottom')

def smacplot(day, showplot = 'no'):
    avg50 = np.array([])
    avg200 = np.array([])
    decision = np.array(['hold'])
    closeThisYear = close[0:day+1][::-1] # closing prices of stocks for all trading days of this year starting from the first day
    for i in range(day+1, 0, -1):
        avg50 = np.append(avg50, np.average(close[i:i+50])) # averages of previous 50 days for all the trading days starting from last trading day of the last year
        avg200 = np.append(avg200, np.average(close[i:i+200])) # averages of previous 200 days for all the trading days starting from last trading day of the last year
    for j in range(1, len(avg50)-1):
        if avg50[j-1] < avg200[j-1] and avg200[j] < avg50[j]:
            decision = np.append(decision, 'buy')
        elif avg50[j-1] > avg200[j-1] and avg200[j] > avg50[j]:
            decision = np.append(decision, 'sell')
        else:
            decision = np.append(decision, 'hold')
    decision = np.append(decision, 'sell')
    if showplot == 'yes':
        d = np.arange(0, 251)
        plotann(decision, d, avg50) # annotate buys and sells
        plt.title('Graph plot of Simple Moving Average')
        plt.xlabel('Trading days of year 2017')
        plt.ylabel('Closing prices, 50 day averages, 200 day averages')
        plt.xlim(min(d), max(d))
        plt.ylim(min(closeThisYear), max(closeThisYear))
        firstDays = [0, 19, 40, 62, 82, 105, 125, 147, 169, 188, 211, 230, 250]
        monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(firstDays, monthNames)
        plt.plot(d, closeThisYear, color = "red",  label = "Closing Year")
        plt.plot(d, avg50, color = "blue", label = "Average 50")
        plt.plot(d, avg200, color = "green", label = "Average 200")
        plt.legend()
        plt.show()
    return decision

def macdplot(day, showplot = 'no'):
    #function that calculates moving average crossover/divergence and creates array of decisions for each day of the error
    decision = np.array(['hold'])
    #count = 0
    emas12 = np.array([np.average(close[day+1:day+13])]) #initialize EMA12 for the first day
    emas26 = np.array([np.average(close[day+1:day+27])]) #initialize EMA26 for the first day
    for i in range(day, 0, -1):
        emas12 = np.append(emas12, (2*close[i]/13) + (11*emas12[-1]/13)) #calculate EMA12 for each day between first day of the year and last day of the year
        emas26 = np.append(emas26, (2*close[i]/27) + (25*emas26[-1]/27)) #calculate EMA12 for each day between first day of the year and last day of the year
    macd = emas12 - emas26 #calculate MACD
    signal = np.array([np.average(macd[0:9])]) #initialize signal for the first day
    for i in range(len(macd)-1):
        signal = np.append(signal, 2*macd[i+1]+8*signal[-1]/10) #calculate signal for each day between first day and last day of the year
    for i in range(len(macd)-1):
        if macd[i] < signal[i] and signal[i+1] < macd[i+1]:
            decision = np.append(decision, 'buy')
        elif macd[i] > signal[i] and signal[i+1] > macd[i+1]:
            decision = np.append(decision, 'sell')
        else:
            decision = np.append(decision, 'hold')
    decision[-1] = 'sell'
    closeThisYear = close[0:day+1][::-1] # closing prices of stocks for all trading days of this year starting from the first day
    if showplot == 'yes':
        d = np.arange(0, 251)
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.title('Moving Average Crossover')
        plt.xlabel('Trading Days of 2017')
        plt.ylabel('Closing Prices and EMA values')
        plt.xlim(min(d), max(d))
        firstDays = [0, 19, 40, 62, 82, 105, 125, 147, 169, 188, 211, 230, 250]
        monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(firstDays, monthNames)
        plt.plot(d, closeThisYear, color = "red",  label = "Closing Year")
        plt.plot(d, emas12, color = "green",  label = "EMA 12")
        plt.plot(d, emas26, color = "blue",  label = "EMA 26")
        plt.legend()
        plt.subplot(2, 1, 2)
        plotann(decision, d, macd) # annotate buys and sells
        plt.xlabel('Trading Days of 2017')
        plt.ylabel('MACD and Signal')
        firstDays = [0, 19, 40, 62, 82, 105, 125, 147, 169, 188, 211, 230, 250]
        monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(firstDays, monthNames)
        plt.xlim(min(d), max(d))
        plt.plot(d, macd, color = "red",  label = "MACD")
        plt.plot(d, signal, color = "green",  label = "Signal")
        plt.legend()
        plt.show()
    return decision

def rsinplot(day, showplot = 'no'):
    #function that calculates rsin and creates array of decisions for each day of the year
    decision = np.array([])
    gainsLosses = (close[:day+1] - open[:day+1])[::-1]
    gainsplot = np.array([])
    lossesplot = np.array([])
    gainsday = np.array([])
    lossesday = np.array([])
    rsiplot = np.array([])
    for j in range(0, len(gainsLosses)):
        if gainsLosses[j] > 0:
            gainsplot = np.append(gainsplot, gainsLosses[j])
            gainsday = np.append(gainsday, j)
        else:
            lossesplot = np.append(lossesplot, abs(gainsLosses[j]))
            lossesday = np.append(lossesday, j)
    for i in range(day+1, 0, -1):
        gains = np.array([])
        losses = np.array([])
        for j in range(14):
            if close[i+j] > open[i+j]:
                gains = np.append(gains, close[i+j]-open[i+j]) #calculate gains
            else:
                losses = np.append(losses, open[i+j]-close[i+j]) #calculate losses
        if len(losses) == 0:
            rsin = 100 #if no losses, rsin = 100
        else:
            avggain = np.average(gains) #get average of gains
            avgloss = np.average(losses) #get average of losses
            rsn = avggain/avgloss #calculate rsn
            rsin = 100-100/(1+rsn) #calculate rsin
            rsiplot = np.append(rsiplot, rsin)
        if rsin < 30:
            decision = np.append(decision, 'buy')
        elif rsin > 70:
            decision = np.append(decision, 'sell')
        else:
            decision = np.append(decision, 'hold')
    decision[0] = 'hold'
    decision[-1] = 'sell'
    d = np.arange(0, 251)
    if showplot == 'yes':
        plt.subplot(2, 1, 1)
        plt.title('Relative Strength Index Plot')
        plt.xlabel('Trading Days of 2017\n')
        plt.ylabel('Gains, Losses')
        plt.xlim(min(gainsLosses), max(gainsLosses))
        plt.xlim(min(d), max(d))
        firstDays = [0, 19, 40, 62, 82, 105, 125, 147, 169, 188, 211, 230, 250]
        monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(firstDays, monthNames)
        plt.bar(gainsday, gainsplot, color = "red", label = "Gains")
        plt.bar(lossesday, lossesplot, color = "blue", label = "Losses")
        plt.legend()
        plt.subplot(2, 1, 2)
        plotann(decision, d, rsiplot) # annotate buys and sells
        plt.xlabel('Trading Days of 2017')
        plt.ylabel('RSI')
        plt.xlim(min(rsiplot), max(rsiplot))
        plt.xlim(min(d), max(d))
        plt.xticks(firstDays, monthNames)
        plt.bar(d, rsiplot, color = "green")
        rsi30 = np.empty(251)
        rsi30.fill(30)
        rsi70 = np.empty(251)
        rsi70.fill(70)
        plt.plot(d, rsi30, color = "blue")
        plt.plot(d, rsi70, color = "red")
        plt.legend()
        plt.show()
    return decision

def obvplot(day, showplot = 'no'):
    #function that calculates obv and creates array of decisions for each day of the year
    volThisYear = volume[:day+1]
    obvArray = np.array([0])
    decision = np.array(['hold'])
    obvThisYrOnly = np.array([])
    for i in range(day+20, 1, -1):
        if close[i] > close[i+1]:
            if len(obvArray) == 0:
                obvArray = np.append(obvArray, 0+volume[i])
            else:
                obvArray = np.append(obvArray, obvArray[-1] + volume[i])
        elif close[i] < close[i+1]:
            obvArray = np.append(obvArray, obvArray[-1] - volume[i])
        else:
            obvArray = np.append(obvArray, obvArray[-1])
    slopeArray = np.array([])
    for i in range(0, len(obvArray)-19):
        slope, abc = np.polyfit(np.arange(0,20), obvArray[i:i+20], 1)
        slopeArray = np.append(slopeArray, slope)
        obvThisYrOnly = np.append(obvThisYrOnly, obvArray[i])
    for i in range(1, len(slopeArray+1)):
        if slopeArray[i-1] >= 0 and slopeArray[i] < 0:
            decision = np.append(decision, 'sell')
        elif slopeArray[i-1] < 0 and slopeArray[i] >= 0:
            decision = np.append(decision, 'buy')
        else:
            decision = np.append(decision, 'hold')
    decision[-1] = 'sell'
    d = np.arange(0, 251)
    if showplot == 'yes':
        fig = plt.figure(1)
        plt.title('On Balance Volume Plot')
        plt.xlabel('Trading Days of 2017\n')
        plt.ylabel('Daily Volumes, OBV')
        plt.xlim(min(volThisYear), max(volThisYear))
        plt.xlim(min(d), max(d))
        firstDays = [0, 19, 40, 62, 82, 105, 125, 147, 169, 188, 211, 230, 250]
        monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(firstDays, monthNames)
        plt.bar(d, volThisYear, color = "red", label = "Volumes This Year")
        plt.bar(d, obvThisYrOnly, color = "blue", label = "OBV This year", alpha = 0.5 )
        plt.legend()
        plotann(decision, d, obvThisYrOnly) # annotate buys and sells
        b = obvThisYrOnly - d * slopeArray
        xline = np.linspace(d[0], d[0]+3, 251)
        yline = slopeArray[0] * xline + b[0]
        lin, = plt.plot(xline, yline, color = 'red')

        def init_line() :
            pass

        def update(frame) :
        
            if d[frame] - 3 < 0:
                xline = np.linspace(0, d[frame] + 3, 50)
            else:
                xline = np.linspace(d[frame] - 3, d[frame] + 3, 50)
        
            yline = slopeArray[frame] * xline + b[frame]
            lin.set_data(xline,yline)
            return lin,


        ani = anim.FuncAnimation(fig, update, frames=251, init_func=init_line, interval = 50, repeat = False)
        plt.show()
    return decision

def mulindi(day):
    #function that takes decisions of all 4 indicators for each day of the year and returns the decisions based on them
    decision = np.array(['hold'])
    for i in range(day, 1, -1):
        eachDayDec = []
        eachDayDec.append(smacplot(day)[i])
        eachDayDec.append(macdplot(day)[i])
        eachDayDec.append(obvplot(day)[i])
        eachDayDec.append(rsinplot(day)[i])
        if eachDayDec.count('buy') >= 2:
            decision = np.append(decision, 'buy')
        elif eachDayDec.count('sell') >= 2:
            decision = np.append(decision, 'sell')
        else:
            decision = np.append(decision, 'hold')
    decision = np.append(decision, 'sell')
    return decision

def getMoney(decisions):
    # function to array with money in the bank for each day of the year based on decisions
    day = 250
    bank = 1000%close[day+1]
    stockHand = (1000-bank)/close[day+1]
    stockBought = 0
    stockSold = 0
    action = ''
    bankArr = np.array([])
    for i in range(len(decisions)):
        if decisions[i] == 'buy':
            if bank > open[day-i]:
                bought = (bank-(bank%open[day-i]))/open[day-i]
                stockHand += bought
                spent = open[day-i]*((bank-(bank%open[day-i]))/open[day-i])
                bank -= spent
        elif decisions[i] == 'sell' and i != day:
            if stockHand > 0:
                earned = stockHand*open[day-i]
                bank += earned
                sold = stockHand
                stockHand = 0
        elif decisions[i] == 'sell' and i == day:
            if stockHand > 0:
                earned = stockHand*open[day-i]
                bank += earned
                sold = stockHand
                stockHand = 0
        bankArr = np.append(bankArr, bank)
    return bankArr

def animImag(arr, arr1, arr2, arr3, arr4):
    # function to plot bar graphs and animate showing the fluctuations of money in the bank for each day of the year for all indicators including the multi-indicators
    d = np.arange(0, 251)
    fig = plt.figure(1)
    plt.title('Bar graph and animation showing money fluctuations for each day of the year')
    plt.xlabel('Trading Days of 2017\n')
    plt.ylabel('$1000 investment fluctuations')
    firstDays = [0, 19, 40, 62, 82, 105, 125, 147, 169, 188, 211, 230, 250]
    monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(firstDays, monthNames)
    plt.bar(d, arr, color = 'red')
    plt.xlim(min(d), max(d))
    xline = np.linspace(d[0], d[0]+3, 251)
    yline = arr[0] * xline
    lin, = plt.plot(xline, yline, color = 'red', label = 'SMAC')
    
    def init_line() :
        pass
        
    def update(frame) :
        if d[frame] - 3 < 0:
            xline = np.linspace(0, d[frame] + 3, 5)
        else:
            xline = np.linspace(d[frame] - 3, d[frame] + 3, 5)
            
        yline = arr[frame]
        lin.set_data(xline,yline)
        return lin,
        
        
    ani = anim.FuncAnimation(fig, update, frames=251, init_func=init_line, interval = 50, repeat = False)

    plt.bar(d, arr1, color = 'blue', alpha = 0.8)
    xline1 = np.linspace(d[0], d[0]+3, 251)
    yline1 = arr1[0] * xline1
    lin1, = plt.plot(xline1, yline1, color = 'blue', label = 'MACD')
    def update1(frame) :
        if d[frame] - 3 < 0:
            xline1 = np.linspace(0, d[frame] + 3, 5)
        else:
            xline1 = np.linspace(d[frame] - 3, d[frame] + 3, 5)
        
        yline1 = arr1[frame]
        lin1.set_data(xline1,yline1)
        return lin1,

    ani1 = anim.FuncAnimation(fig, update1, frames=251, init_func=init_line, interval = 50, repeat = False)
    
    plt.bar(d, arr2, color = 'green', alpha = 0.6)
    xline2 = np.linspace(d[0], d[0]+3, 251)
    yline2 = arr2[0] * xline2
    lin2, = plt.plot(xline2, yline2, color = 'green', label = 'RSI')
    def update2(frame) :
        if d[frame] - 3 < 0:
            xline2 = np.linspace(0, d[frame] + 3, 5)
        else:
            xline2 = np.linspace(d[frame] - 3, d[frame] + 3, 5)
        
        yline2 = arr2[frame]
        lin2.set_data(xline2,yline2)
        return lin2,
    
    ani2 = anim.FuncAnimation(fig, update2, frames=251, init_func=init_line, interval = 50, repeat = False)
    
    plt.bar(d, arr3, color = 'black', alpha = 0.4)
    xline3 = np.linspace(d[0], d[0]+3, 251)
    yline3 = arr3[0] * xline3
    lin3, = plt.plot(xline3, yline3, color = 'black', label = 'OBV')
    
    def update3(frame) :
        if d[frame] - 3 < 0:
            xline3 = np.linspace(0, d[frame] + 3, 5)
        else:
            xline3 = np.linspace(d[frame] - 3, d[frame] + 3, 5)
        
        yline3 = arr3[frame]
        lin3.set_data(xline3,yline3)
        return lin3,
    
    ani3 = anim.FuncAnimation(fig, update3, frames=251, init_func=init_line, interval = 50, repeat = False)

    plt.bar(d, arr4, color = 'orange', alpha = 0.2)
    xline4 = np.linspace(d[0], d[0]+3, 251)
    yline4 = arr4[0] * xline4
    lin4, = plt.plot(xline4, yline4, color = 'orange', label = 'Multiple index')
    
    def update4(frame) :
        if d[frame] - 3 < 0:
            xline4 = np.linspace(0, d[frame] + 3, 5)
        else:
            xline4 = np.linspace(d[frame] - 3, d[frame] + 3, 5)
        
        yline4 = arr4[frame]
        lin4.set_data(xline4,yline4)
        return lin4,
    
    ani4 = anim.FuncAnimation(fig, update4, frames=251, init_func=init_line, interval = 50, repeat = False)
    
    
    plt.legend()
    plt.show()

def imagination(day):
    smacMoney = getMoney(smacplot(day)) # get money fluctuations array of SMAC
    macMoney = getMoney(macdplot(day)) # get money fluctuations array of MACD
    rsiMoney = getMoney(rsinplot(day)) # get money fluctuations array of RSI
    obvMoney = getMoney(obvplot(day)) # get money fluctuations array of OBV
    multiMoney = getMoney(mulindi(day)) # get money fluctuations array of multi-idicators
    animImag(smacMoney, macMoney, rsiMoney, obvMoney, multiMoney) # show all animations showing $1000 fluctuations for all idicators

def p2plot():
    day = 250  #250 is the first trading day of 2017
    while True:
        option = input('Enter 1 for SMAC Plot\nEnter 2 for MACD Plot\nEnter 3 for RSI Plot\nEnter 4 for OBV Plot\nEnter 5 to show $1000 investments fluctuations\nEnter 6 to quit\nEnter your option:')
        if option == '1':
            smacplot(day, showplot = 'yes')
        elif option == '2':
            macdplot(day, showplot = 'yes')
        elif option == '3':
            rsinplot(day, showplot = 'yes')
        elif option == '4':
            obvplot(day, showplot = 'yes')
        elif option == '5':
            imagination(day)
        elif option == '6':
            break
        else:
            print('invalid option, try again.')
            continue

p2plot()


