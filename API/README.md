# API Usage

URL - ```<https://crypto-project-api.herokuapp.com/>```

----
***/get_data/single/market/index/date***

*Type - GET*

Sample Request - 

```https://crypto-project-api.herokuapp.com/get_data/single/crypto/bitcoin/2020-12-05```

Sample Respose - 

```
{
  "data":
  [
    {
      "close":19154.23046875,
      "date":"2020-12-05",
      "high":19160.44921875,
      "low":18590.193359375,
      "open":18698.384765625
    }
  ],
  "status":"Success"
}
```

----

***/get_data/multiple/market/index/start_date/end_date***

*Type - GET*

Sample Request - 

```https://crypto-project-api.herokuapp.com/get_data/multiple/crypto/bitcoin/2020-12-02/2020-12-05```

Sample Respose - 

```
{
  "data":
  [
    {
      "close":"19201.091796875",
      "date":"2020-12-02",
      "high":"19308.330078125",
      "low":"18347.71875",
      "open":"18801.744140625"
    },
    {
      "close":"19371.041015625",
      "date":"2020-12-03",
      "high":"19430.89453125",
      "low":"18937.4296875",
      "open":"18949.251953125"
    },
    {
      "close":19154.23046875,
      "date":"2020-12-05",
      "high":19160.44921875,
      "low":18590.193359375,
      "open":18698.384765625
    }
  ],
  "status":"Success"
}
```
----

***/get_predictions/date***

*Type - GET*

Sample Request - 

```https://crypto-project-api.herokuapp.com/get_predictions/2020-12-05```

Sample Respose - 

```
{
  "data":
    [
      {
        "bitcoin":"16204.04",
        "dash":"24.148237",
        "date":"2020-12-05",
        "ethereum":"503.43005",
        "litecoin":"66.6938",
        "monero":"120.718414",
        "ripple":"0.55850273"
      }
    ],
  "status":"Success"
}
```
