import pandas as pd
from tcn_model import *
from util import *
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    df = pd.read_csv("stock_simulation.csv") # Generate by SDE models
    train_df, test_df = train_test_split(df, test_size = 0.2, shuffle = False)

    n_stocks   = df.shape[1] - 1  # drop timestamp
    input_len  = 60
    output_len = 5

    train_ds = TimeSeriesDataset(train_df, input_len, output_len)
    train_dl = DataLoader(train_ds, batch_size = 32, shuffle = True)

    test_ds = TimeSeriesDataset(test_df, input_len, output_len)
    test_dl = DataLoader(test_ds, batch_size = 32, shuffle = True)

    model = TCNForecastModel(
        input_dim    = n_stocks,
        hidden_dims  = [64, 64, 64],
        output_steps = output_len,
    ).to("cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    criterion = nn.MSELoss()

    for epoch in range(1, 51):
        loss = train_one_epoch(model, train_dl, optimizer, criterion, "cpu")
        test_loss = evaluate(model, test_dl, criterion, "cpu")
        print(f"Epoch {epoch:03d} | Train Loss = {loss:.4f} | Validation Loss = {test_loss:.4f}")

