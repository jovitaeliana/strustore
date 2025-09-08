# Zenmarket Scraper (mini)

This README is written to clear up any confusion with the naming conventions I've used and why I've got so much crap (lol sorry).

## Context

This scraper was created when GroundingDINO needed to be trained and Yahoo Japan Auction (yjpa) scraper was found to be pulling listings with its poorer english translation in its search engine compared to zenmarket. As the data was gathered for training and testing purposes of an object detection model, the scraping was manually grouped into 5 family of items, defined in `items.json`.  

The families are grouped by their item characteristics. And I later continued to keep the files in these folders labelled 1 to 5 for no further purpose, just the sake of referencing and simple organization.

## Files and Information

The files created and referenced below act as a "database" to this scraper without actually using a database.

- `items.json`: defines item families within
- `master-list.csv`: individual items grouped into families 
- `items-url.csv`: All urls gathered for each family to be scraped
- `items-prompt.csv`: Hardcoded prompts used in groundingdino for each family.

## Assets
 
All assets are located inside the `zm_scraper_assets` folder of Drive.

Use SCP to move files from your local workstation to the instance via SSH.
```
scp -i ~/.ssh/yourkeyhere -r "./Downloads/yourfile.zip" user@ip:./strustore-auction-ai/auctions/
```

Asset List:
- raw: raw images from zenmarket
- preprocessed: raw images with backgrounds removed using remove.bg, saved in the [preview](https://www.remove.bg/lt/help/a/what-is-the-maximum-image-resolution-file-size) version
- gdino: all files that has been run through the gdino route from preprocessed to the end of the classification pipeline.