# utils
import pandas as pd
from utils import *
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
# keras/TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dropout, GRU, Bidirectional, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam

# set random seed
np.random.seed(123)

train_file = "../dataset/datachallenge-traindata.csv"
df_train = pd.read_csv(train_file, sep=';')
# df_train.head()

seqs = df_train["seq"].values
X = build_onehot_matrix(seqs, padd=True)
y = df_train['label']
lb = LabelBinarizer()
y = lb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# specify number of filters and filter width
n_filters = 20
filter_width = 7
# sequence length
seq_len = X_train.shape[1]

gru_model = Sequential()
gru_model.add(Conv1D(n_filters, filter_width, activation='relu', input_shape=(seq_len, 4)))
gru_model.add(MaxPooling1D(name='pool'))
gru_model.add(Dropout(0.2))
gru_model.add(Bidirectional(GRU(128, activation='relu')))
gru_model.add(BatchNormalization())
gru_model.add(Dropout(0.2))
gru_model.add(Dense(64, activation='relu'))
gru_model.add(Dense(1, activation='sigmoid'))

gru_model.compile(optimizer=Adam(learning_rate=1e-5, decay=1e-6), loss='binary_crossentropy', metrics=['accuracy'])
# specify number of epochs
n_epochs = 150
# fit linear model
history_lin = gru_model.fit(X_train, y_train, batch_size=128, epochs=n_epochs, verbose=1, validation_split=0.2)

# compute train performance
pred_train = gru_model.predict(X_train)
pred_train = pred_train >0.5
pred_train = pred_train*1
# compute confusion matrix
cm = confusion_matrix(y_train, pred_train)
print("\n**** confusion matrix ****")
print(cm)
# compute sensi/speci and macro accuracy
sensi = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity: ', sensi )

speci = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity: ', speci)

macro_acc = 0.5*(sensi+speci)
print('Macro accuracy: ', macro_acc)
