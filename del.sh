#!/bin/bash

TARGET_DIR="./raw/gugak" 
KEEP_COUNT=300

if [ ! -d "$TARGET_DIR" ]; then
    echo "cannot find folder: $TARGET_DIR"
    exit 1
fi

cd "$TARGET_DIR" || exit

TOTAL_FILES=$(ls -1 *.wav 2>/dev/null | wc -l)
TOTAL_FILES=$((TOTAL_FILES + 0))

echo "current items on '$TARGET_DIR': $TOTAL_FILES"

if [ "$TOTAL_FILES" -le "$KEEP_COUNT" ]; then
    echo "already lower than $KEEP_COUNT. "
    exit 0
fi

ls -1 *.wav | sort -R | head -n "$DELETE_COUNT" | while read -r file; do
    rm "$file"
done

echo "after remove items: $(ls -1 *.wav 2>/dev/null | wc -l)"