try:
    from tkinter import *
    from tkinter import ttk
    from ttkthemes import ThemedTk
    import numpy as np  # linear algebra
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pandas import DataFrame
    from keras.utils.np_utils import to_categorical  # one hot encod
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
    from keras.optimizers import adam_v2
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import LambdaCallback
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    errorC = False
except:
    errorC = True
if(errorC == False):
    mainW = ThemedTk(theme="equilux", themebg=True, toplevel=True)
    mainW.title("CNN Tester v 1.1 (Mnist Dataset)")
    mainWFX = 1920
    mainWFY = 1080
    mainW.geometry(str(mainWFX)+"x"+str(mainWFY))
    ttkstyle = ttk.Style()
    ttkstyle.configure('Red.TLabelframe.Label',
                       font=("TkDefaultFont", 12, "bold"))
    oran = IntVar()
    labelframe = ttk.LabelFrame(
        mainW, text="Settings", style="Red.TLabelframe")
    labelframe.place(x=0, y=0, height=1150, width=650)
    labelframe1 = ttk.LabelFrame(labelframe, text="Random stade:")
    labelframe1.grid(sticky=W, row=3, column=1)
    labelframe2 = ttk.LabelFrame(labelframe, text="Max pool size:")
    labelframe2.grid(sticky=W, row=3, column=3)
    labelframe3 = ttk.LabelFrame(labelframe, text="Optimizer learning rate:")
    labelframe3.grid(sticky=W, row=2, column=3, padx=0, pady=10)
    labelframe4 = ttk.LabelFrame(labelframe, text="Number of Epochs:")
    labelframe4.grid(sticky=W, row=3, column=2)
    labelframe5 = ttk.LabelFrame(labelframe, text="Batch Size:")
    labelframe5.grid(sticky=W, row=2, column=1, padx=0, pady=10)
    labelframe6 = ttk.LabelFrame(labelframe, text="Datagenerate range:")
    labelframe6.grid(sticky=W, row=2, column=2, padx=0, pady=10)
    labelframe7 = ttk.LabelFrame(
        labelframe, text="Convolution Layers Settings:", style="Red.TLabelframe")
    labelframe7.place(x=0, y=150, height=250, width=450)
    labelframe8 = ttk.LabelFrame(
        labelframe, text="Fully Connected Layers Settings:", style="Red.TLabelframe")
    labelframe8.place(x=0, y=450, height=350, width=450)
    label1 = ttk.Label(labelframe, text="v 1.1")
    label1.place(x=0, y=930)
    label2 = ttk.Label(labelframe, text="osmankocakank@gmail.com")
    label2.place(x=0, y=950)
    labelframe9 = ttk.LabelFrame(labelframe8, text="FCL layer 1 Functions:")
    labelframe9.grid(sticky=W, row=0, column=0, padx=0, pady=10)
    labelframe10 = ttk.LabelFrame(labelframe8, text="FCL layer 2 Functions:")
    labelframe10.grid(sticky=W, row=0, column=1, padx=0, pady=10)
    labelframe11 = ttk.LabelFrame(labelframe8, text="FCL layer 3 Functions:")
    labelframe11.grid(sticky=W, row=0, column=2, padx=0, pady=10)
    labelframe12 = ttk.LabelFrame(labelframe8, text="FCL layer 1 Dense:")
    labelframe12.grid(sticky=W, row=1, column=0, padx=0, pady=0)
    labelframe13 = ttk.LabelFrame(labelframe8, text="FCL layer 2 Dense:")
    labelframe13.grid(sticky=W, row=1, column=1, padx=0, pady=0)
    labelframe14 = ttk.LabelFrame(labelframe8, text="FCL layer 3 Dense:")
    labelframe14.grid(sticky=W, row=1, column=2, padx=0, pady=0)
    labelframe15 = ttk.LabelFrame(labelframe8, text="FCL layer 1 Dropout:")
    labelframe15.grid(sticky=W, row=2, column=0, padx=0, pady=0)
    labelframe16 = ttk.LabelFrame(labelframe8, text="FCL layer 2 Dropout:")
    labelframe16.grid(sticky=W, row=2, column=1, padx=0, pady=0)
    labelframe17 = ttk.LabelFrame(labelframe8, text="FCL layer 3 Dropout:")
    labelframe17.grid(sticky=W, row=2, column=2, padx=0, pady=0)
    labelframe18 = ttk.LabelFrame(labelframe8, text="Output layer Functions:")
    labelframe18.grid(sticky=W, row=3, column=0, padx=0, pady=0)
    labelframe19 = ttk.LabelFrame(labelframe7, text="CNL layer 1 Functions:")
    labelframe19.grid(sticky=W, row=0, column=0, padx=0, pady=10)
    labelframe20 = ttk.LabelFrame(labelframe7, text="CNL layer 2 Functions:")
    labelframe20.grid(sticky=W, row=0, column=1, padx=0, pady=10)
    labelframe21 = ttk.LabelFrame(labelframe7, text="CNL layer 3 Functions:")
    labelframe21.grid(sticky=W, row=0, column=2, padx=0, pady=10)
    labelframe22 = ttk.LabelFrame(labelframe7, text="CNL layer 1 Filters:")
    labelframe22.grid(sticky=W, row=1, column=0, padx=0, pady=0)
    labelframe23 = ttk.LabelFrame(labelframe7, text="CNL layer 2 Filters:")
    labelframe23.grid(sticky=W, row=1, column=1, padx=0, pady=0)
    labelframe24 = ttk.LabelFrame(labelframe7, text="CNL layer 3 Filters:")
    labelframe24.grid(sticky=W, row=1, column=2, padx=0, pady=0)
    labelframe25 = ttk.LabelFrame(labelframe7, text="CNL layer 1 Dropout:")
    labelframe25.grid(sticky=W, row=2, column=0, padx=0, pady=0)
    labelframe26 = ttk.LabelFrame(labelframe7, text="CNL layer 2 Dropout:")
    labelframe26.grid(sticky=W, row=2, column=1, padx=0, pady=0)
    labelframe27 = ttk.LabelFrame(labelframe7, text="CNL layer 3 Dropout:")
    labelframe27.grid(sticky=W, row=2, column=2, padx=0, pady=0)

    btntrain = ttk.Button(mainW, text="TRAIN", command=lambda: readData())
    btntrain.place(x=650, y=10, height=1070, width=100)

    btndef = ttk.Button(mainW, text="Default Settings",
                        command=lambda: setdefsettings())
    btndef.place(x=0, y=835, height=75, width=451)

    progressb = ttk.Progressbar(mainW, orient=VERTICAL,
                                length=1070, mode='determinate')
    progressb.place(x=750, y=10)

    varsCmb1 = ttk.Combobox(labelframe1)
    varsCmb1.grid(sticky=W)
    varsCmb1['values'] = ('2')
    varsCmb1.current(0)
    varsCmb1['state'] = 'readonly'

    varsCmb2 = ttk.Combobox(labelframe2)
    varsCmb2.grid(sticky=W)
    varsCmb2['values'] = ('2x2', '3x3')
    varsCmb2.current(0)
    varsCmb2['state'] = 'readonly'

    varsCmb3 = ttk.Combobox(labelframe3)
    varsCmb3.grid(sticky=W)
    varsCmb3['values'] = ('0.001', '0.01', '0.1', '0.002', '0.02', '0.2')
    varsCmb3.current(0)
    varsCmb3['state'] = 'readonly'
    varsCmb4 = ttk.Combobox(labelframe4)
    varsCmb4.grid(sticky=W)
    varsCmb4['values'] = ('2', '5', '10', '15', '20', '30', '50')
    varsCmb4.current(0)
    varsCmb4['state'] = 'readonly'
    varsCmb5 = ttk.Combobox(labelframe5)
    varsCmb5.grid(sticky=W)
    varsCmb5['values'] = ('32', '64', '128', '256')
    varsCmb5.current(0)
    varsCmb5['state'] = 'readonly'
    varsCmb6 = ttk.Combobox(labelframe6)
    varsCmb6.grid(sticky=W)
    varsCmb6['values'] = ('0.1', '0.2', '0.3', '0.4', '0.5', '0.6')
    varsCmb6.current(0)
    varsCmb6['state'] = 'readonly'

    varsCmb7 = ttk.Combobox(labelframe19)
    varsCmb7.grid(sticky=W)
    varsCmb7['values'] = ("relu", "softmax", "tanh", "sigmoid", "LeakyReLU")
    varsCmb7.current(0)
    varsCmb7['state'] = 'readonly'

    varsCmb14 = ttk.Combobox(labelframe20)
    varsCmb14.grid(sticky=W)
    varsCmb14['values'] = ("disable layer", "relu", "softmax",
                           "tanh", "sigmoid", "LeakyReLU")

    varsCmb14.current(0)
    varsCmb14['state'] = 'readonly'
    varsCmb9 = ttk.Combobox(labelframe21)
    varsCmb9.grid(sticky=W)
    varsCmb9['values'] = ("disable layer", "relu", "softmax",
                          "tanh", "sigmoid", "LeakyReLU")
    varsCmb9.current(0)
    varsCmb9['state'] = 'readonly'
    varsCmb10 = ttk.Combobox(labelframe22)
    varsCmb10.grid(sticky=W, row=1, column=0)
    varsCmb10['values'] = ('8', '16', '32', '64', '128', '256', '512')
    varsCmb10.current(0)
    varsCmb10['state'] = 'readonly'
    varsCmb11 = ttk.Combobox(labelframe23)
    varsCmb11.grid(sticky=W, row=1, column=1)
    varsCmb11['values'] = ('8', '16', '32', '64', '128', '256', '512')
    varsCmb11.current(0)
    varsCmb11['state'] = 'readonly'
    varsCmb12 = ttk.Combobox(labelframe24)
    varsCmb12.grid(sticky=W, row=1, column=2)
    varsCmb12['values'] = ('8', '16', '32', '64', '128', '256', '512')
    varsCmb12.current(0)
    varsCmb12['state'] = 'readonly'
    varsCmb13 = ttk.Combobox(labelframe25)
    varsCmb13.grid(sticky=W, row=2, column=0)
    varsCmb13['values'] = ('0.0', '0.25', '0.50', '0.70')
    varsCmb13.current(0)
    varsCmb13['state'] = 'readonly'
    varsCmb14 = ttk.Combobox(labelframe26)
    varsCmb14.grid(sticky=W, row=2, column=1)
    varsCmb14['values'] = ('0.0', '0.25', '0.50', '0.70')
    varsCmb14.current(0)
    varsCmb14['state'] = 'readonly'
    varsCmb15 = ttk.Combobox(labelframe27)
    varsCmb15.grid(sticky=W, row=2, column=2)
    varsCmb15['values'] = ('0.0', '0.25', '0.50', '0.70')
    varsCmb15.current(0)
    varsCmb15['state'] = 'readonly'
    ####
    varsCmb16 = ttk.Combobox(labelframe9)
    varsCmb16['values'] = ("relu", "softmax", "tanh", "sigmoid", "LeakyReLU")
    varsCmb16.grid(sticky=W)
    varsCmb16.current(0)
    varsCmb16['state'] = 'readonly'
    varsCmb17 = ttk.Combobox(labelframe10)
    varsCmb17.grid(sticky=W)
    varsCmb17['values'] = ("disable layer", "relu", "softmax",
                           "tanh", "sigmoid", "LeakyReLU")
    varsCmb17.current(0)
    varsCmb17['state'] = 'readonly'
    varsCmb18 = ttk.Combobox(labelframe11)
    varsCmb18.grid(sticky=W)
    varsCmb18['values'] = ("disable layer", "relu", "softmax",
                           "tanh", "sigmoid", "LeakyReLU")
    varsCmb18.current(0)
    varsCmb18['state'] = 'readonly'
    varsCmb19 = ttk.Combobox(labelframe12)
    varsCmb19.grid(sticky=W)
    varsCmb19['values'] = ('8', '16', '32', '64', '128', '256', '512')
    varsCmb19.current(0)
    varsCmb19['state'] = 'readonly'
    varsCmb20 = ttk.Combobox(labelframe13)
    varsCmb20.grid(sticky=W)
    varsCmb20['values'] = ('8', '16', '32', '64', '128', '256', '512')
    varsCmb20.current(0)
    varsCmb20['state'] = 'readonly'
    varsCmb21 = ttk.Combobox(labelframe14)
    varsCmb21.grid(sticky=W)
    varsCmb21['values'] = ('8', '16', '32', '64', '128', '256', '512')
    varsCmb21.current(0)
    varsCmb21['state'] = 'readonly'
    varsCmb22 = ttk.Combobox(labelframe15)
    varsCmb22.grid(sticky=W)
    varsCmb22['values'] = ('0.0', '0.25', '0.50', '0.70')
    varsCmb22.current(0)
    varsCmb22['state'] = 'readonly'
    varsCmb23 = ttk.Combobox(labelframe16)
    varsCmb23.grid(sticky=W)
    varsCmb23['values'] = ('0.0', '0.25', '0.50', '0.70')
    varsCmb23.current(0)
    varsCmb23['state'] = 'readonly'
    varsCmb24 = ttk.Combobox(labelframe17)
    varsCmb24.grid(sticky=W)
    varsCmb24['values'] = ('0.0', '0.25', '0.50', '0.70')
    varsCmb24.current(0)
    varsCmb24['state'] = 'readonly'
    varsCmb25 = ttk.Combobox(labelframe18)
    varsCmb25.grid(sticky=W)
    varsCmb25['values'] = ("relu", "softmax", "tanh", "sigmoid", "LeakyReLU")
    varsCmb25.current(0)
    varsCmb25['state'] = 'readonly'
    ####
    v_random_state = int(varsCmb1.get())
else:
    EmainW = Tk()
    EmainW.title("Report any issue to: osmankocakank@gmail.com")
    EmainW.geometry("450x25")
    EmainW.resizable(False, False)
    text = "Something went wrong!"
    lbl = Label(
        EmainW, text=text, fg="Red")
    lbl.grid(sticky="W", row=0, column=0)


def getHypervars(X_train, Y_train, X_test, Y_test, X_val, Y_val, orgX_Test, orgY_Test):
    # hyper param variables
    v_filters_cnn_l1 = int(varsCmb10.get())
    v_filters_cnn_l2 = int(varsCmb11.get())
    v_filters_cnn_l3 = int(varsCmb12.get())
    v_dense_l1 = int(varsCmb19.get())
    v_dense_l2 = int(varsCmb20.get())
    v_dense_l3 = int(varsCmb21.get())
    v_dropouts_cnn_l1 = float(varsCmb13.get())
    v_dropouts_cnn_l2 = float(varsCmb14.get())
    v_dropouts_cnn_l3 = float(varsCmb15.get())
    v_dropouts_dense_l1 = float(varsCmb22.get())
    v_dropouts_dense_l2 = float(varsCmb23.get())
    v_dropouts_dense_l3 = float(varsCmb24.get())
    v_activation_functions_cnn_l1 = varsCmb7.get()
    v_activation_functions_cnn_l2 = varsCmb14.get()
    v_activation_functions_cnn_l3 = varsCmb9.get()
    v_activation_functions_dense_l1 = varsCmb16.get()
    v_activation_functions_dense_l2 = varsCmb17.get()
    v_activation_functions_dense_l3 = varsCmb18.get()
    v_activation_functions_output = varsCmb25.get()
    v_optimizer_lr = float(varsCmb3.get())
    v_epochs = int(varsCmb4.get())
    v_batch_size = int(varsCmb5.get())
    v_datagenerate_range = float(varsCmb6.get())
    createModel(X_train, Y_train, X_test, Y_test, v_filters_cnn_l1, v_filters_cnn_l2, v_filters_cnn_l3, v_dense_l1, v_dense_l2, v_dense_l3, v_dropouts_cnn_l1, v_dropouts_cnn_l2, v_dropouts_cnn_l3, v_dropouts_dense_l1, v_dropouts_dense_l2, v_dropouts_dense_l3, v_activation_functions_cnn_l1, v_activation_functions_cnn_l2,
                v_activation_functions_cnn_l3, v_activation_functions_dense_l1, v_activation_functions_dense_l2, v_activation_functions_dense_l3,
                v_activation_functions_output, v_optimizer_lr, v_epochs, v_batch_size, v_datagenerate_range, X_val, Y_val, orgX_Test)


def setdefsettings():
    varsCmb10.current(0)
    varsCmb11.current(0)
    varsCmb12.current(0)
    varsCmb19.current(0)
    varsCmb20.current(0)
    varsCmb21.current(0)
    varsCmb13.current(0)
    varsCmb14.current(0)
    varsCmb15.current(0)
    varsCmb22.current(0)
    varsCmb23.current(0)
    varsCmb24.current(0)
    varsCmb7.current(0)
    varsCmb14.current(0)
    varsCmb9.current(0)
    varsCmb16.current(0)
    varsCmb17.current(0)
    varsCmb18.current(0)
    varsCmb25.current(0)
    varsCmb3.current(0)
    varsCmb4.current(0)
    varsCmb5.current(0)
    varsCmb6.current(0)
    mainW.update_idletasks()


def maxPoolSize():
    if(varsCmb2.get() == "2x2"):
        v_Maxpoolsize_xy = 2
    elif(varsCmb2.get() == "3x3"):
        v_Maxpoolsize_xy = 3
    else:
        pass
    return v_Maxpoolsize_xy


def readData():
    train = pd.read_csv("mnist_train.csv")
    print("train:", train.shape)

    Y_train = train["label"]
    X_train = train.drop(labels=["label"], axis=1)
    print("xtrain:", X_train.shape)
    print("Ytrain:", Y_train.shape)

    test = pd.read_csv("mnist_test.csv")
    Y_test = test["label"]
    X_test = test.drop(labels=["label"], axis=1)
    print("Xtest:", X_test.shape)
    print("firstYtest:", Y_test.shape)

    test = test.drop(labels=["label"], axis=1)

    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    X_test = X_test.values.reshape(-1, 28, 28, 1)

    print("xtrain shape:", X_train.shape)
    print("xtest laleli shape:", X_test.shape)

    orgX_Test = X_test

    Y_train = to_categorical(Y_train, num_classes=10)
    Y_test = to_categorical(Y_test, num_classes=10)
    print("ytest laleli shape:", Y_test.shape)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=v_random_state)
    print("xtrain shape", X_train.shape)
    print("xval shape", X_val.shape)
    print("ytrain shape", Y_train.shape)
    print("yval shape", Y_val.shape)

    getHypervars(X_train, Y_train, X_test, Y_test,
                 X_val, Y_val, orgX_Test)


def createModel(X_train, Y_train, X_test, Y_test, v_filters_cnn_l1, v_filters_cnn_l2, v_filters_cnn_l3, v_dense_l1, v_dense_l2, v_dense_l3, v_dropouts_cnn_l1, v_dropouts_cnn_l2, v_dropouts_cnn_l3, v_dropouts_dense_l1, v_dropouts_dense_l2, v_dropouts_dense_l3, v_activation_functions_cnn_l1, v_activation_functions_cnn_l2,
                v_activation_functions_cnn_l3, v_activation_functions_dense_l1, v_activation_functions_dense_l2, v_activation_functions_dense_l3,
                v_activation_functions_output, v_optimizer_lr, v_epochs, v_batch_size, v_datagenerate_range, X_val, Y_val, orgX_Test):
    model = Sequential()

    # /cnn_l1
    model.add(Conv2D(filters=v_filters_cnn_l1, kernel_size=(5, 5), padding="Same",
                     activation=v_activation_functions_cnn_l1, input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(maxPoolSize(), maxPoolSize())))
    model.add(Dropout(v_dropouts_cnn_l1))
    # cnn_l1/
    # /cnn_l2
    if(varsCmb14.get() != "disable layer"):
        model.add(Conv2D(filters=v_filters_cnn_l2, kernel_size=(5, 5),
                         padding="Same", activation=v_activation_functions_cnn_l2))
        model.add(MaxPool2D(pool_size=(maxPoolSize(), maxPoolSize())))
        model.add(Dropout(v_dropouts_cnn_l2))
    else:
        pass
    # cnn_l2/
    # /cnn_l3
    if(varsCmb9.get() != "disable layer"):
        model.add(Conv2D(filters=v_filters_cnn_l3, kernel_size=(5, 5),
                         padding="Same", activation=v_activation_functions_cnn_l3))
        model.add(MaxPool2D(pool_size=(maxPoolSize(), maxPoolSize())))
        model.add(Dropout(v_dropouts_cnn_l3))
    else:
        pass
    # cnn_l3/

    model.add(Flatten())

    model.add(Dense(v_dense_l1, activation=v_activation_functions_dense_l1))
    model.add(Dropout(v_dropouts_dense_l1))
    if(varsCmb17.get() != "disable layer"):
        model.add(Dense(v_dense_l2, activation=v_activation_functions_dense_l2))
        model.add(Dropout(v_dropouts_dense_l2))
    else:
        pass
    if(varsCmb18.get() != "disable layer"):
        model.add(Dense(v_dense_l3, activation=v_activation_functions_dense_l3))
        model.add(Dropout(v_dropouts_dense_l3))
    else:
        pass
    model.add(Dense(10, activation=v_activation_functions_output))

    optimizerM = adam_v2.Adam(lr=v_optimizer_lr, beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=optimizerM,
                  loss="categorical_crossentropy", metrics=["accuracy"])

    epochs = v_epochs
    batch_size = v_batch_size

    datagenerate = ImageDataGenerator(
        featurewise_center=False,  # def
        samplewise_center=False,  # def
        featurewise_std_normalization=False,  # def
        samplewise_std_normalization=False,  # def
        zca_whitening=False,  # def
        rotation_range=v_datagenerate_range,  # can change
        zoom_range=v_datagenerate_range,  # can change
        width_shift_range=v_datagenerate_range,  # can change
        height_shift_range=v_datagenerate_range,  # can change
        horizontal_flip=False,  # can change
        vertical_flip=False  # can change
    )

    datagenerate.fit(X_train)

    trainFitModel(model, datagenerate, X_train, Y_train,
                  batch_size, epochs, X_test, Y_test, X_val, Y_val, orgX_Test)


def pbar():
    pstr = ""
    x = oran
    if(progressb['value'] >= x):
        progressb['value'] = x+progressb['value']
        pstr = str(round(progressb['value'], 1))
    elif(progressb['value'] == 0.0):
        pstr = str(round(progressb['value'], 1))
        progressb['value'] = 0.1
    else:
        progressb['value'] = x
        pstr = str(round(progressb['value'], 1))
    value = progressb['value']
    print("SON value", round(progressb['value'], 1))
    ttk.Label(mainW, text="%"+pstr).place(x=683, y=555)
    mainW.update_idletasks()


def pfinished():
    progressb['value'] = 100
    ttk.Label(mainW, text="%"+str(progressb['value'])).place(x=683, y=555)
    mainW.update_idletasks()
    progressb['value'] = 0


def callback():
    global epoch_callback, train_finish_callback
    epoch_callback = LambdaCallback(on_epoch_begin=lambda epoch, logs: pbar())
    train_finish_callback = LambdaCallback(
        on_train_end=lambda logs: pfinished()
    )


def trainFitModel(model, datagenerate, X_train, Y_train, batch_size, epochs, X_test, Y_test, X_val, Y_val, orgX_Test):
    callback()
    global oran
    x = 100 / epochs
    oran = x
    history = model.fit_generator(datagenerate.flow(X_train, Y_train, batch_size=batch_size),
                                  epochs=epochs, validation_data=(X_test, Y_test), steps_per_epoch=X_train.shape[0] // batch_size, callbacks=[epoch_callback, train_finish_callback])
    df = DataFrame(history.history['loss'])
    df1 = DataFrame(history.history['val_loss'])
    figure1 = plt.Figure(figsize=(6, 5), dpi=100)
    ax1 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, mainW)
    bar1.get_tk_widget().place(x=765, y=10)
    df.plot(kind='line', legend=True, ax=ax1)
    df1.plot(kind='line', legend=True, ax=ax1)
    ax1.set_title('Test Loss')
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(["Train loss", "Test loss"], loc=1)

    confMatrix(model, X_val, Y_val)
    pltaccuracy(history)
    imgshow(X_val, Y_val, model, orgX_Test)


def confMatrix(model, X_val, Y_val):
    Y_pred = model.predict(X_val)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_val, axis=1)
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    figure2, ax2 = plt.subplots(figsize=(6, 5))
    df2 = sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,
                      cmap="Greens", linecolor="gray", fmt='.1f', ax=ax2)
    bar2 = FigureCanvasTkAgg(figure2, mainW)
    bar2.get_tk_widget().place(x=1350, y=10)
    df2.plot(kind='line', legend=True, ax=ax2)
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")


def imgshow(X_test, Y_test, model, orgX_Test):
    predicted_classes = model.predict(orgX_Test[0:9])
    y_pred_cls = np.argmax(predicted_classes, axis=1)
    y_true_cls = np.argmax(Y_test[0:9], axis=1)
    #incorrect_indices = np.nonzero(y_pred_cls != y_true_cls)[0]
    fig, axes = plt.subplots(3, 3, figsize=(6, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.gray()
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[i].reshape(28, 28))
        xlabel = "True: {0}, Pred: {1}".format(y_true_cls[i], y_pred_cls[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    bar2 = FigureCanvasTkAgg(fig, mainW)
    bar2.get_tk_widget().place(x=1350, y=500)


def pltaccuracy(history):
    df3 = DataFrame(history.history['accuracy'])
    df4 = DataFrame(history.history['val_accuracy'])
    figure3 = plt.Figure(figsize=(6, 5), dpi=100)
    ax3 = figure3.add_subplot(111)
    bar3 = FigureCanvasTkAgg(figure3, mainW)
    bar3.get_tk_widget().place(x=765, y=500)
    df3.plot(kind='line', legend=True, ax=ax3)
    df4.plot(kind='line', legend=True, ax=ax3)
    ax3.set_title('model accuracy')
    ax3.set_xlabel('Number of Epochs')
    ax3.set_ylabel('accuracy')
    ax3.legend(["Train accuracy", "Test accuracy"], loc=1)


# combo box vars
cmbx_var_random_state = StringVar()
cmbx_var_filters = StringVar()
cmbx_var_dense = StringVar()
cmbx_var_Maxpoolsize_x = StringVar()
cmbx_var_Maxpoolsize_y = StringVar()
cmbx_var_dropouts = StringVar()
cmbx_var_activation_functions = StringVar()
cmbx_var_optimizer_lr = StringVar()
cmbx_var_epochs = StringVar()
cmbx_var_batch_size = StringVar()
cmbx_var_datagenerate_range = StringVar()
#
mainloop()
