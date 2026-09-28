| # | Question | Gold SQL | Predicted SQL | Correct |
|---|---|---|---|---|
| 1 | What school/club team is Amir Johnson on? | `SELECT School/Club Team FROM table WHERE Player = 'Amir Johnson'` | `SELECT School/Club Team FROM table WHERE Player = 'amir johnson'` | yes |
| 2 | What is the highest Rd that Tom Sneva had the pole position in? | `SELECT MAX(Rd) FROM table WHERE Pole Position = 'Tom Sneva'` | `SELECT MAX(Rd) FROM table WHERE Pole Position = 'tom sneva'` | yes |
| 3 | What was the score of the game on November 12? | `SELECT Score FROM table WHERE Date = 'November 12'` | `SELECT Score FROM table WHERE Date = 'november 12'` | yes |
| 4 | Who had high assists when they played against San Antonio? | `SELECT High assists FROM table WHERE Team = 'San Antonio'` | `SELECT High assists FROM table WHERE Team = 'san antonio'` | yes |
| 5 | Which team was the second semi finalist in 2007? | `SELECT Semi-Finalist #2 FROM table WHERE Year = '2007'` | `SELECT Semi-Finalist #2 FROM table WHERE Year = '2007'` | yes |
| 6 | What position does the player who played for butler cc (ks) play? | `SELECT Position FROM table WHERE School/Club Team = 'Butler CC (KS)'` | `SELECT No. FROM table WHERE Player = 'butler cc'` | no |
| 7 | How many schools did player number 3 play at? | `SELECT COUNT(School/Club Team) FROM table WHERE No. = '3'` | `SELECT COUNT(School/Club Team) FROM table WHERE Player = '3'` | no |
| 8 | What school did player number 21 play for? | `SELECT School/Club Team FROM table WHERE No. = '21'` | `SELECT No. FROM table WHERE Years in Toronto = '21'` | no |
| 9 | Who is the player that wears number 42? | `SELECT Player FROM table WHERE No. = '42'` | `SELECT No. FROM table WHERE Position = '42'` | no |
| 10 | What player played guard for toronto in 1996-97? | `SELECT Player FROM table WHERE Position = 'Guard' AND Years in Toronto = '1996-97'` | `SELECT Player FROM table WHERE School/Club Team = 'guard' AND Years in Toronto = '1996-97'` | no |
