import random

# Базовые параметры
DAYS = 5
PERIODS = 4
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
    """Подсчитывает количество конфликтов в расписании"""
    conflicts = 0
    
    for day in range(DAYS):
        for period in range(PERIODS):
            # Проверка повторения преподавателей или аудиторий
            teachers_used = {}
            rooms_used = {}
            
            for group in range(GROUPS):
                subject, teacher, room = timetable[day][period][group]
                
                if teacher in teachers_used:
                    conflicts += 1
                teachers_used[teacher] = True
                
                if room in rooms_used:
                    conflicts += 1
                rooms_used[room] = True
    
    return conflicts

def create_initial_schedule():
    """Создает начальное случайное расписание"""
    timetable = []
    
    for day in range(DAYS):
        day_schedule = []
        for period in range(PERIODS):
            period_schedule = []
            for group in range(GROUPS):
                subject, teacher = random.choice(subjects_teachers)
                room = random.randint(1, ROOMS)
                period_schedule.append((subject, teacher, room))
            day_schedule.append(period_schedule)
        timetable.append(day_schedule)
    
    return timetable

def optimize_schedule(iterations=100):
    """Оптимизирует расписание методом локального поиска"""
    current = create_initial_schedule()
    current_conflicts = calculate_conflicts(current)
    
    best = current
    best_conflicts = current_conflicts
    
    for i in range(iterations):
        # Выбираем случайную позицию для изменения
        day = random.randint(0, DAYS-1)
        period = random.randint(0, PERIODS-1)
        group = random.randint(0, GROUPS-1)
        
        # Создаем копию текущего расписания
        neighbor = [[[current[d][p][g] for g in range(GROUPS)] for p in range(PERIODS)] for d in range(DAYS)]
        
        # Изменяем выбранную позицию
        subject, teacher = random.choice(subjects_teachers)
        room = random.randint(1, ROOMS)
        neighbor[day][period][group] = (subject, teacher, room)
        
        # Оцениваем новое расписание
        neighbor_conflicts = calculate_conflicts(neighbor)
        
        # Принимаем изменение, если оно улучшает расписание или с вероятностью
        if neighbor_conflicts < current_conflicts or random.random() < 0.1:
            current = neighbor
            current_conflicts = neighbor_conflicts
            
            # Обновляем лучшее найденное решение
            if current_conflicts < best_conflicts:
                best = current
                best_conflicts = current_conflicts
    
    return best, best_conflicts

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
final_schedule, conflicts = optimize_schedule(200)

print(f"Итоговое расписание (конфликтов: {conflicts})")
print_schedule(final_schedule)

