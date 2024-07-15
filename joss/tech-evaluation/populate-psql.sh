#!/bin/bash

# Set variables
CSV_DIR=$1
DB_NAME="public"

# Hard code database user and password
# Easily modify to get them from environment variables
DB_USER="postgres"
DB_PASSWORD="postgres"

# Create the PostgreSQL database if it does not exist
psql -U $DB_USER -c "CREATE DATABASE $DB_NAME TEMPLATE template0 ENCODING 'UTF8' LC_COLLATE 'en_US.UTF-8' LC_CTYPE 'en_US.UTF-8';" > /dev/null 2>&1

# Loop through each CSV file in the directory
# This is the 20M records csv files
for csv_file in "$CSV_DIR"/*.csv; do

  # Read the first line of the CSV file to get the column headers
  IFS=',' read -ra HEADERS <<< $(head -n 1 "$csv_file")

  # Create the table in the public schema of the new database if it does not exist
  TABLE_NAME=$(basename "$csv_file" | sed 's/-/_/g' | sed 's/ /_/g' | cut -d. -f1)
  # drop if exists
  DROP_TABLE=$(printf "DROP TABLE IF EXISTS public.%s;" "$TABLE_NAME")
  psql -U $DB_USER -d $DB_NAME -c "$DROP_TABLE"
  # create the table
  CREATE_TABLE=$(printf "CREATE TABLE IF NOT EXISTS public.%s (%s);" "$TABLE_NAME" "$(echo "${HEADERS[@]}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//; s/ / text, /g; s/@//g') text")
  psql -U $DB_USER -d $DB_NAME -c "$CREATE_TABLE"

  # Copy the CSV data into the new table 5 times
  for i in {1..5}
  do
    psql -U $DB_USER -d $DB_NAME -c "\copy public.$TABLE_NAME FROM '$csv_file' DELIMITER ',' CSV HEADER"
  done

  # Add a primary key column called 'key'
  ALTER_TABLE=$(printf "ALTER TABLE public.%s ADD COLUMN key SERIAL PRIMARY KEY;" "$TABLE_NAME")
  psql -U $DB_USER -d $DB_NAME -c "$ALTER_TABLE"

  # Show number of rows inserted
  ROW_COUNT=$(psql -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM public.$TABLE_NAME")
  echo "Inserted $ROW_COUNT rows into table $TABLE_NAME in database $DB_NAME."
done
