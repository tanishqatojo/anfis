import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os

class ANFIS:
    def __init__(self, n_inputs, n_rules, n_epochs, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        
        self.a = np.random.randn(n_inputs, n_rules)
        self.b = np.random.randn(n_inputs, n_rules)
        self.c = np.random.randn(n_inputs, n_rules)
        
        self.p = np.random.randn(n_rules, n_inputs + 1)
        
    def membership_function(self, x, a, b, c):
        epsilon = 1e-10
        a = np.maximum(a, epsilon) 

        base = (x - c) / a
        base = np.maximum(base, 0)

        return 1 / (1 + base ** (2 * b))
    
    def forward_pass(self, X):
        mf_values = np.array([[self.membership_function(x, self.a[i, j], self.b[i, j], self.c[i, j])
                               for j in range(self.n_rules)] for i, x in enumerate(X)])
        
        w = np.prod(mf_values, axis=0)
        
        w_norm = w / (np.sum(w) + 1e-10)
        
        f = np.dot(self.p, np.append(X, 1))
        
        y = np.sum(w_norm * f)
        
        return y
    
    def train(self, X, y):
        for epoch in range(self.n_epochs):
            total_loss = 0
            for i in range(len(X)):
                pred = self.forward_pass(X[i])
                target_class = np.argmax(y[i]) 
                error = target_class - pred 
            
                total_loss += error ** 2
            
                self.p += self.learning_rate * error * np.append(X[i], 1)
                self.a += self.learning_rate * error * np.random.randn(*self.a.shape)
                self.b += self.learning_rate * error * np.random.randn(*self.b.shape)
                self.c += self.learning_rate * error * np.random.randn(*self.c.shape)
        
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(X)}")
    
    def predict(self, X):
        return np.array([self.forward_pass(x) for x in X])

if __name__ == "__main__":
    directory = "/Users/lilmoltojo/Desktop/archive (5)" 
    
    benign = pd.read_csv(os.path.join(directory, '5.benign.csv'))
    g_c = pd.read_csv(os.path.join(directory, '5.gafgyt.combo.csv'))
    g_j = pd.read_csv(os.path.join(directory, '5.gafgyt.junk.csv'))
    g_s = pd.read_csv(os.path.join(directory, '5.gafgyt.scan.csv'))
    g_t = pd.read_csv(os.path.join(directory, '5.gafgyt.tcp.csv'))
    g_u = pd.read_csv(os.path.join(directory, '5.gafgyt.udp.csv'))
    m_a = pd.read_csv(os.path.join(directory, '5.mirai.ack.csv'))
    m_sc = pd.read_csv(os.path.join(directory, '5.mirai.scan.csv'))
    m_sy = pd.read_csv(os.path.join(directory, '5.mirai.syn.csv'))
    m_u = pd.read_csv(os.path.join(directory, '5.mirai.udp.csv'))
    m_u_p = pd.read_csv(os.path.join(directory, '5.mirai.udpplain.csv'))

    benign = benign.sample(frac=0.25, replace=False)
    g_c = g_c.sample(frac=0.25, replace=False)
    g_j = g_j.sample(frac=0.5, replace=False)
    g_s = g_s.sample(frac=0.5, replace=False)
    g_t = g_t.sample(frac=0.15, replace=False)
    g_u = g_u.sample(frac=0.15, replace=False)
    m_a = m_a.sample(frac=0.25, replace=False)
    m_sc = m_sc.sample(frac=0.15, replace=False)
    m_sy = m_sy.sample(frac=0.25, replace=False)
    m_u = m_u.sample(frac=0.1, replace=False)
    m_u_p = m_u_p.sample(frac=0.27, replace=False)

    benign['type'] = 'benign'
    m_u['type'] = 'mirai_udp'
    g_c['type'] = 'gafgyt_combo'
    g_j['type'] = 'gafgyt_junk'
    g_s['type'] = 'gafgyt_scan'
    g_t['type'] = 'gafgyt_tcp'
    g_u['type'] = 'gafgyt_udp'
    m_a['type'] = 'mirai_ack'
    m_sc['type'] = 'mirai_scan'
    m_sy['type'] = 'mirai_syn'
    m_u_p['type'] = 'mirai_udpplain'

    data = pd.concat([benign, m_u, g_c, g_j, g_s, g_t, g_u, m_a, m_sc, m_sy, m_u_p], axis=0, sort=False, ignore_index=True)

    print(data.groupby('type')['type'].count())


    sampler = np.random.permutation(len(data))
    data = data.take(sampler)

    labels_full = pd.get_dummies(data['type'], prefix='type')
    data = data.drop(columns='type')

    X = data.values 
    y = labels_full.values  
    print("Shape of X:", X.shape)  
    print("Shape of y:", y.shape)  

    def standardize(df, col):
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    data_st = data.copy()
    for i in data_st.columns:
        standardize(data_st, i)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

#ANFIS model
    n_inputs = X_train.shape[1]
    anfis = ANFIS(n_inputs=n_inputs, n_rules=5, n_epochs=100, learning_rate=0.01)
    anfis.train(X_train_scaled, y_train)

    train_predictions = anfis.predict(X_train_scaled)
    test_predictions = anfis.predict(X_test_scaled)

    train_predictions_binary = (train_predictions > 0.5).astype(int)
    test_predictions_binary = (test_predictions > 0.5).astype(int)

    train_accuracy = accuracy_score(y_train.argmax(axis=1), train_predictions_binary)
    test_accuracy = accuracy_score(y_test.argmax(axis=1), test_predictions_binary)
    
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test.argmax(axis=1), test_predictions_binary))
