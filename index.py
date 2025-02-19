import random

# Базовые параметры
DAYS = 5
PERIODS = 3
GROUPS = 4
ROOMS = 4

# Предметы и преподаватели
subjects_teachers = [
    ("Data Structure and Algorithms", "Askarov K.R"),
    ("English", "Абакирова Э.А"),
    ("Introduction to AI", "Beishenalieva A."),
    ("Advanced Python", "Prof. Daechul Park"),
    ("География Кыргызстана", "Жумалиев Н.Э."),
    ("История Кыргызстана", "Молошев А.И."),
    ("Манасоведение", "Бегалиев Э.С."),
]

def calculate_conflicts(timetable):
    """Подсчитывает количество конфликтов в расписании с повышенным вниманием к учителям"""
    conflicts = 0
    
    for day in range(DAYS):
        for period in range(PERIODS):
            # Проверка повторения преподавателей или аудиторий
            teachers_used = {}
            rooms_used = {}
            
            for group in range(GROUPS):
                subject, teacher, room = timetable[day][period][group]
                
                # Серьезный конфликт - один учитель в разных местах одновременно
                if teacher in teachers_used:
                    conflicts += 10  # Увеличиваем штраф за учителей
                teachers_used[teacher] = True
                
                if room in rooms_used:
                    conflicts += 2
                rooms_used[room] = True
    
    return conflicts

def create_initial_schedule():
    """Создает начальное расписание с учетом доступности преподавателей"""
    timetable = []
    
    for day in range(DAYS):
        day_schedule = []
        for period in range(PERIODS):
            # Для каждого периода отслеживаем, какие преподаватели уже заняты
            available_teachers = {teacher for _, teacher in subjects_teachers}
            period_schedule = []
            
            for group in range(GROUPS):
                # Выбираем только из доступных преподавателей
                possible_courses = [(subject, teacher) for subject, teacher in subjects_teachers 
                                   if teacher in available_teachers]
                
                # Если нет доступных преподавателей, откладываем этот слот
                if not possible_courses:
                    # Выбираем любой курс, конфликт будет разрешен позже
                    subject, teacher = random.choice(subjects_teachers)
                else:
                    subject, teacher = random.choice(possible_courses)
                    available_teachers.remove(teacher)
                
                room = random.randint(1, ROOMS)
                period_schedule.append((subject, teacher, room))
            
            day_schedule.append(period_schedule)
        timetable.append(day_schedule)
    
    return timetable

def fix_teacher_conflicts(timetable):
    """Исправляет конфликты преподавателей путем замены предметов"""
    for day in range(DAYS):
        for period in range(PERIODS):
            # Отслеживаем занятых преподавателей в текущем периоде
            teachers_assigned = {}
            conflicting_groups = []
            
            # Сначала находим конфликты
            for group in range(GROUPS):
                subject, teacher, room = timetable[day][period][group]
                
                if teacher in teachers_assigned:
                    conflicting_groups.append(group)
                else:
                    teachers_assigned[teacher] = True
            
            # Исправляем найденные конфликты
            for group in conflicting_groups:
                current_subject, current_teacher, current_room = timetable[day][period][group]
                
                # Ищем альтернативный предмет с доступным преподавателем
                alternative_options = [(s, t) for s, t in subjects_teachers 
                                      if t not in teachers_assigned]
                
                if alternative_options:
                    # Если есть альтернатива, используем ее
                    new_subject, new_teacher = random.choice(alternative_options)
                    timetable[day][period][group] = (new_subject, new_teacher, current_room)
                    teachers_assigned[new_teacher] = True
                else:
                    # Если нет доступных преподавателей, меняем местами с другим периодом
                    # Находим другой период, где этот преподаватель не занят
                    alternative_periods = []
                    for other_period in range(PERIODS):
                        if other_period == period:
                            continue
                            
                        teachers_in_period = {timetable[day][other_period][g][1] for g in range(GROUPS)}
                        if current_teacher not in teachers_in_period:
                            alternative_periods.append(other_period)
                    
                    if alternative_periods:
                        swap_period = random.choice(alternative_periods)
                        # Меняем местами занятия для этой группы
                        timetable[day][period][group], timetable[day][swap_period][group] = \
                            timetable[day][swap_period][group], timetable[day][period][group]
    
    return timetable

def assign_rooms_without_conflicts(timetable):
    """Назначает аудитории без конфликтов"""
    for day in range(DAYS):
        for period in range(PERIODS):
            rooms_assigned = set()
            
            for group in range(GROUPS):
                subject, teacher, _ = timetable[day][period][group]
                
                # Выбираем свободную аудиторию
                available_rooms = set(range(1, ROOMS + 1)) - rooms_assigned
                if available_rooms:
                    room = random.choice(list(available_rooms))
                else:
                    # Если все аудитории заняты, это будет отмечено как конфликт
                    room = random.randint(1, ROOMS)
                
                rooms_assigned.add(room)
                timetable[day][period][group] = (subject, teacher, room)
    
    return timetable

def optimize_schedule(iterations=200):
    """Оптимизирует расписание методом локального поиска с исправлением конфликтов"""
    current = create_initial_schedule()
    current = fix_teacher_conflicts(current)
    current = assign_rooms_without_conflicts(current)
    
    current_conflicts = calculate_conflicts(current)
    
    best = current
    best_conflicts = current_conflicts
    
    for i in range(iterations):
        # Делаем копию текущего расписания
        neighbor = [[[current[d][p][g] for g in range(GROUPS)] for p in range(PERIODS)] for d in range(DAYS)]
        
        # Выбираем две случайные группы для обмена предметами
        day = random.randint(0, DAYS-1)
        period = random.randint(0, PERIODS-1)
        group1 = random.randint(0, GROUPS-1)
        group2 = random.randint(0, GROUPS-1)
        
        # Меняем предметы местами
        if group1 != group2:
            neighbor[day][period][group1], neighbor[day][period][group2] = \
                neighbor[day][period][group2], neighbor[day][period][group1]
        
        # После обмена исправляем возможные конфликты преподавателей
        neighbor = fix_teacher_conflicts(neighbor)
        neighbor = assign_rooms_without_conflicts(neighbor)
        
        # Оцениваем новое расписание
        neighbor_conflicts = calculate_conflicts(neighbor)
        
        # Принимаем изменение, если оно улучшает расписание
        if neighbor_conflicts < current_conflicts:
            current = neighbor
            current_conflicts = neighbor_conflicts
            
            # Обновляем лучшее найденное решение
            if current_conflicts < best_conflicts:
                best = neighbor
                best_conflicts = neighbor_conflicts
    
    # Финальное исправление любых оставшихся конфликтов
    best = fix_teacher_conflicts(best)
    best = assign_rooms_without_conflicts(best)
    
    return best, calculate_conflicts(best)

def verify_no_teacher_conflicts(timetable):
    """Проверяет отсутствие конфликтов преподавателей и возвращает список проблем"""
    conflicts = []
    
    for day in range(DAYS):
        for period in range(PERIODS):
            teachers_seen = {}
            
            for group in range(GROUPS):
                subject, teacher, room = timetable[day][period][group]
                
                if teacher in teachers_seen:
                    conflicts.append(f"День {day+1}, Период {period+1}: {teacher} назначен группам {teachers_seen[teacher]} и {group+1}")
                else:
                    teachers_seen[teacher] = group+1
    
    return conflicts

def print_schedule(timetable):
    """Выводит расписание в читаемом формате"""
    for day in range(DAYS):
        print(f"День {day+1}")
        print("=" * 50)
        
        for period in range(PERIODS):
            print(f"Период {period+1}:")
            
            for group in range(GROUPS):
                subject, teacher, room = timetable[day][period][group]
                print(f"  Группа {group+1}: {subject} ({teacher}) - Аудитория {room}")
            
            print("-" * 30)
        print()

# Запуск оптимизации
print("Оптимизация расписания...")
final_schedule, conflicts = optimize_schedule(300)

# Проверка на отсутствие конфликтов преподавателей
teacher_conflicts = verify_no_teacher_conflicts(final_schedule)
if teacher_conflicts:
    print("ВНИМАНИЕ! Найдены конфликты преподавателей:")
    for conflict in teacher_conflicts:
        print(f"  - {conflict}")
    print("Повторное исправление конфликтов...")
    final_schedule = fix_teacher_conflicts(final_schedule)
    final_schedule = assign_rooms_without_conflicts(final_schedule)

# Финальная проверка
final_teacher_conflicts = verify_no_teacher_conflicts(final_schedule)
if final_teacher_conflicts:
    print("После исправления все еще остались конфликты преподавателей.")
else:
    print("Расписание без конфликтов преподавателей успешно создано!")

print(f"\nИтоговое расписание (конфликтов: {calculate_conflicts(final_schedule)})")
print_schedule(final_schedule)