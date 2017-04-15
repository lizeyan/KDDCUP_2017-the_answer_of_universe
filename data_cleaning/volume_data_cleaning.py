__author__ = 'Victor'

file_object = open('volume_cleaning_tmp.csv')
output_text = ""

last_tidir = {}
last_tidir['1,0'] = '3,1'
last_tidir['1,1'] = '1,0'
last_tidir['2,0'] = '1,1'
last_tidir['3,0'] = '2,0'
last_tidir['3,1'] = '3,0'

def isnear(timenr, lastimenr):
    if timenr - lastimenr == 1:
        return True
    return False

def date_to_day(date):  # date-int
    date -= 17062
    day = date % 7
    return day

def make_empty_line(td, timenr, date): # td-string, timenr-int, date-int
    emptyline = td
    emptyline += ",0,0,"
    time = timenr * 1200
    emptyline += str(time) + ","
    emptyline += str(timenr) + ","
    emptyline += str(date_to_day(date)) + ","
    emptyline += str(date) + "\n"
    return emptyline


def make_empty_lines_from_to(td1, timenr1, td2, timenr2, date1, date2):  # 设timenr1与timenr2处于同一天
    emptylines = []
    ret = ""
    ldate = date2
    ltimenr = timenr2
    ltd = last_tidir[td2]
    if ltd == "3,1":
        ltimenr -= 1
        if ltimenr == -1:
            ltimenr = 71
            ldate -= 1

    while ltd != td1 or timenr1 != ltimenr or date1 != ldate:
        emptylines.append(make_empty_line(ltd, ltimenr, ldate))
        ltd = last_tidir[ltd]
        if ltd == "3,1":
            ltimenr -= 1
            if ltimenr == -1:
                ltimenr = 71
                ldate -= 1

    emptylines.reverse()
    for line in emptylines:
        ret += line
    return ret

# e = make_empty_lines_from_to("3,1", 59, "1,0", 0, 17063, 17064)
# print(e)



try:
    all_the_text = file_object.read().splitlines()

    lastline = ""
    cnt = 0
    for line in all_the_text:
        cnt += 1
        print(cnt)
        # print(line)
        if cnt > 5:
            info = line.split(',')
            last_info = lastline.split(',')

            date = int(info[7])
            last_date = int(last_info[7])

            if int(info[6]) != date_to_day(date):
                newline = ""
                for i in range(0, 6):
                    newline += info[i] + ","
                newline += str(date_to_day(date)) + ","
                newline += info[7]
                line = newline

            timenr = int(info[5])
            last_timenr = int(last_info[5])

            td = line[0:3]
            last_td = lastline[0:3]

            output_text += make_empty_lines_from_to(last_td, last_timenr, td, timenr, last_date, date)


        lastline = line

        output_text += line + "\n"

finally:
    file_object.close()
    file_object = open('volume_cleaning_result.csv', 'w')
    file_object.write(output_text)
    file_object.close()
