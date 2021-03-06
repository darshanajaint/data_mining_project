#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:32:35 2021

@author: ithier
"""

import nest_asyncio
import twint
import pandas as pd
import numpy as np
import argparse
import unicodedata

def get_tweets_for_handle(df, csv_row):
    print("--------------------------")
    print("Collecting data for ", csv_row.loc[0, "Twitter Handle"])
    
    nest_asyncio.apply()
    
    c = twint.Config()
    c.Username = csv_row.loc[0, "Twitter Handle"]
    c.Since = "2020-01-01"
    c.Pandas = True
    twint.run.Search(c)
    
    tweets_df = twint.storage.panda.Tweets_df
    tweets_df = tweets_df[["date", "tweet", "hashtags", "link", "urls"]]
    
    df_to_add = pd.DataFrame(np.repeat(csv_row.values, len(tweets_df), axis=0), columns=csv_row.columns)
    
    df_to_add = pd.concat([tweets_df, df_to_add], axis=1)
    
    df = pd.concat([df, df_to_add], ignore_index=True)
    
    return df

def get_tweets(args):
    # df of the csv with information about senators
    senators_df = pd.read_csv(args.senators_csv)
    
    # df of senator info along with tweets
    df = pd.DataFrame(columns=senators_df.columns)
    
    for i in range(len(senators_df)):
        row = senators_df.iloc[[i]]
        row = row.reset_index(drop=True)
        if row.loc[0, "Multiple"] == 0:
            df = get_tweets_for_handle(df, row)
            df.to_csv(args.output_file, index=False)
        else:
            handles =row.loc[0, "Twitter Handle"].split(",")
            for handle in handles:
                row.loc[0, "Twitter Handle"] = handle.strip()
                df = get_tweets_for_handle(df, row)
                df.to_csv(args.output_file, index=False)
    
    df.to_csv(args.output_file, index=False)
    
    print("-----------------------------")
    print("Wrote file to ", args.output_file)

def main():
    parser = argparse.ArgumentParser(description='Collect tweets from senators')
    parser.add_argument('--senators_csv', type=str, 
                        help='csv file that has senator names, handles, party, etc')
    parser.add_argument('--output_file', type=str,
                        help='output csv filename')
    
    args = parser.parse_args()
    get_tweets(args)
    
main()


