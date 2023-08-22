import Data_Parser_pcap as d_p
import Model_Utils as m_u
import Models as mo

if __name__ == '__main__':
    y, x = d_p.load_parsed_datasets()
    print(x)
    x_tr, x_te, y_tr, y_te = m_u.load_data_to_trainable_dataset(x=x, y=y)
    naive_bayes_model = mo.create_naive_bayes_model(6)
    model_history = mo.train_model(model=naive_bayes_model, x_train=x_tr, y_train=y_tr, x_val=x_te, y_val=y_te, epochs=25)