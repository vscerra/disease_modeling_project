"""
===========================================================
measles_ab_model.py
Author: Veronica Scerra
Last Updated: 2026-01-28
===========================================================
Agent-Based Model for Measles Transmission in Schools
======================================================

Module implements an individual-based SEIR model for measles
transmission within a school (controlled) setting. The model
captures: 
- Heterogeneous contact patterns w/in vs between classrooms
- Spatial clustering by grade/classroom
- Vaccination coverage with realistic coverage patterns
- Stochastic transmission dynamics
- Individual-level disease progression

License: MIT
===========================================================
"""
import numpy as np 
from dataclasses import dataclass 
from typing import List, Tuple, Dict, Optional 
from enum import Enum 

class DiseaseState(Enum):
    """Enumeration for SEIR disease states"""
    SUSCEPTIBLE = 0
    EXPOSED = 1
    INFECTIOUS = 2
    RECOVERED = 3

@dataclass 
class MeaslesParameters:
    """Epidemiological parameters for measles transmission
    Attributes:
    beta_within: float. Transmission prob per contact within classroom (daily)
    beta_between: float. Transmission prob per contact between classrooms (daily)
    sigma: float. Rate of progression from Exposed to Infectious (1/latent_period)
    gamma: float. Rate of progression from Infectious to Recovered (1/infectious_period)
    contacts_within: int. Average number of daily contacts within classroom 
    contacts_between: int. Avg number of daily contacts outside classroom
    vaccine_efficiency: float. Probability that vaccine provides protection

    Notes: 
    Default values based on measles epidemiology:
    - latent_period: ~10 days (sigma = 0.1)
    - infectious_period: ~8 days (gamma = 0.125)
    - R0 in fully susceptible population: typically 12-18
    """
    beta_within: float = 0.08 
    beta_between: float = 0.005
    sigma: float = 0.1  # 1/10 days latent period
    gamma: float = 0.125  # 1/8 days infectious period
    contacts_within: int = 20
    contacts_between: int = 5
    vaccine_efficacy: float = 0.97
    
    @property
    def R0_within(self) -> float:
        """Basic reproduction number for within-classroom transmission"""
        return self.beta_within * self.contacts_within / self.gamma
    
    @property
    def R0_between(self) -> float:
        """Basic reproduction number for between-classroom transmission"""
        return self.beta_between * self.contacts_between / self.gamma
    
class Student:
    """Individual agent representing student in the school.
    Attributes:
    id: int. Unique identifier
    grade: int. Grade level (1-6 for elementary school)
    classroom: int. Classroom assignment within grade
    state: DiseaseState. Current disease state (S, E, I, R)
    vaccinated: bool. Whether student has been vaccinated 
    protected: bool. Whether student is protected (vaccinated and immune)
    time_in_state: float. Days spent in current disease state
    infection_time: float. Simulation day when infection occurred (if applicable)
    """
    def __init__(self, id: int, grade: int, classroom: int, 
                 vaccinated: bool = False, protected: bool = False):
        self.id = id
        self.grade = grade
        self.classroom = classroom
        self.state = DiseaseState.RECOVERED if protected else DiseaseState.SUSCEPTIBLE
        self.vaccinated = vaccinated
        self.protected = protected
        self.time_in_state = 0.0
        self.infection_time = None
        
    def expose(self, current_time: float):
        """Transition from Susceptible to Exposed"""
        if self.state == DiseaseState.SUSCEPTIBLE and not self.protected:
            self.state = DiseaseState.EXPOSED
            self.time_in_state = 0.0
            self.infection_time = current_time
            
    def progress_disease(self, params: MeaslesParameters, dt: float = 1.0):
        """Progress disease state based on time spent in current state.
        Uses stochastic transitions based on exponentially-distributed durations.
        Parameters:
        params : MeaslesParameters. Model parameters including progression rates
        dt : float. Time step (default 1 day)
        """
        self.time_in_state += dt
        
        if self.state == DiseaseState.EXPOSED:
            # Probability of progressing to infectious
            prob_progress = 1 - np.exp(-params.sigma * dt)
            if np.random.random() < prob_progress:
                self.state = DiseaseState.INFECTIOUS
                self.time_in_state = 0.0
                
        elif self.state == DiseaseState.INFECTIOUS:
            # Probability of recovering
            prob_recover = 1 - np.exp(-params.gamma * dt)
            if np.random.random() < prob_recover:
                self.state = DiseaseState.RECOVERED
                self.time_in_state = 0.0

class School:
    """Representation of a school with grade-based classroom structure
    Attributes:
    n_grades: int. Number of grade levels
    students_per_classroom: int. Number of students in each classroom 
    classrooms_per_grade: int. Number of parallel classrooms per grade 
    students: List[Student]. List of all students in the school    
    """
    def __init__(self, n_grades: int = 6, 
                 students_per_classroom: int = 25,
                 classrooms_per_grade: int = 3):
        self.n_grades = n_grades
        self.students_per_classroom = students_per_classroom
        self.classrooms_per_grade = classrooms_per_grade
        self.students: List[Student] = []
        self.total_students = n_grades * classrooms_per_grade * students_per_classroom
        
        # initialize students
        student_id = 0
        for grade in range(1, n_grades + 1):
            for classroom in range(classrooms_per_grade):
                for _ in range(students_per_classroom):
                    student = Student(student_id, grade, classroom)
                    self.students.append(student)
                    student_id += 1
    
    def vaccinate_population(self, coverage: float, efficacy: float,
                            by_classroom: bool=False, 
                            clustering_factor: float=0.0):
        """Vaccinate students according to specified coverage level
        Parameters:
        coverage: float. Overall fraction of students to vaccinate (0-1)
        efficacy: float. Vaccine efficacy (fraction protected among vaccinated)
        by_classroom: bool. If True, apply coverage at classroom level (creates clustering)
        clustering_factor: float. Degree of clustering (0=random, 1=perfect clustering)
        Notes:
        Clustering simulates real-world patterns where vaccination rates vary by classroom
        due to community attitudes, access, etc. 
        """
        if by_classroom and clustering_factor > 0:
            # create variable coverage by classroom
            classrooms = {}
            for student in self.students:
                key = (student.grade, student.classroom)
                if key not in classrooms:
                    classrooms[key] = []
                classrooms[key].append(student) 

            # assign classroom-level coverage
            for classroom_students in classrooms.values():
                #vary coverage around mean based on clustering factor
                classroom_coverage = np.random.beta(
                    coverage * (1/clustering_factor - 1),
                    (1 - coverage) * (1/clustering_factor - 1)
                )
                classroom_coverage = np.clip(classroom_coverage, 0, 1)

                # vaccinate students in this classroom 
                n_vaccinate = int(len(classroom_students) * classroom_coverage)
                to_vaccinate = np.random.choice(
                    classroom_students, n_vaccinate, replace=False
                )

                for student in to_vaccinate:
                    student.vaccinated = True 
                    if np.random.random() < efficacy:
                        student.protected = True 
                        student.state = DiseaseState.RECOVERED
        else:
            # Random vaccination across population
            n_vaccinate = int(self.total_students * coverage)
            to_vaccinate = np.random.choice(
                self.students, n_vaccinate, replace=False
            )
            
            for student in to_vaccinate:
                student.vaccinated = True
                if np.random.random() < efficacy:
                    student.protected = True
                    student.state = DiseaseState.RECOVERED

    def introduce_infection(self, n_infections: int = 1,
                            target_grade: Optional[int] = None):
        """Introduce initial infections into the school
        Parameters:
        n_infections: int. Number of initial infections to introduce
        target_grade: int, optional. If specified, introduce infections only in this grade
        """
        susceptible = [s for s in self.students
                       if s.state == DiseaseState.SUSCEPTIBLE]
        
        if target_grade is not None:
            susceptible = [s for s in self.grade == target_grade]
        
        if len(susceptible) < n_infections:
            raise ValueError(f"Not enough susceptible students. "
                             f"Requested {n_infections}, found {len(susceptible)}")
        
        initial_infected = np.random.choice(
            susceptible, n_infections, replace=False
        )

        for student in initial_infected:
            student.expose(current_time=0.0)
            # immediately move to infectious for index cases
            student.state = DiseaseState.INFECTIOUS
            student.time_in_state = 0.0 

    def get_counts_by_classroom(self) -> Dict[Tuple[int, int], Dict[str, int]]:
        """Get disease state counts organized by (grade, classroom)"""
        classroom_counts = {}

        for student in self.students:
            key = (student.grade, student.classroom)
            if key not in classroom_counts:
                classroom_counts[key] = {'S': 0, 'E': 0, 'I': 0, 'R': 0}

            if student.state == DiseaseState.SUSCEPTIBLE:
                classroom_counts[key]['S'] += 1
            elif student.state == DiseaseState.EXPOSED:
                classroom_counts[key]['E'] += 1
            elif student.state == DiseaseState.INFECTIOUS:
                classroom_counts[key]['I'] += 1
            elif student.state == DiseaseState.RECOVERED:
                classroom_counts[key]['R'] += 1
                
        return classroom_counts 
    
class MeaslesABM:
    """Agent-based model simulator for measles transmission in schools.
    This class handles the simulation dynamics including:
    - Contact selection within and between classrooms 
    - Stochastic transmission events
    - Disease progression for all agents
    - Temporal tracking of epidemic dynamics
    """
    def __init__(self, school: School, params: MeaslesParameters):
        self.school = school
        self.params = params
        self.current_time = 0.0
        self.history = []

    def simulate_contacts(self, infectious_student: Student) -> List[Student]:
        """Generate contact list for an infectious student 
        Parameters:
        infectious_student: Student. The infectious individual making contacts

        Returns: 
        contacts: List[Student]. List of students contacted during this time step
        """
        contacts = []

        # within-classroom contacts
        n_within = np.random.poisson(self.params.contacts_within)
        same_classroom = [
            s for s in self.school.students 
            if (s.grade == infectious_student.grade and 
                s.classroom == infectious_student.classroom and
                s.id != infectious_student.id and 
                s.state == DiseaseState.SUSCEPTIBLE)
        ]

        n_within = min(n_within, len(same_classroom))
        if n_within > 0:
            within_contacts = np.random.choice(
                same_classroom, n_within, replace=False
            )
            contacts.extend(within_contacts)

        # between-classroom contacts (within same grade and other grades)
        n_between = np.random.poisson(self.params.contacts_between)
        other_students = [
            s for s in self.school.students 
            if (s.id != infectious_student.id and 
                not (s.grade == infectious_student.grade and 
                     s.classroom == infectious_student.classroom) and
                s.state == DiseaseState.SUSCEPTIBLE)
        ]

        n_between = min(n_between, len(other_students))
        if n_between > 0:
            between_contacts = np.random.choice(
                other_students, n_between, replace=False 
            )
            contacts.extend(between_contacts)
        
        return contacts 
    
    def transmission_step(self):
        """Execute one time-step of transmission dynamics.
        for each infectious individual:
        1. Generate contacts (within and between classrooms)
        2. Attempt transmission to each susceptible contact 
        3. Update newly exposed individuals
        """
        infectious = [s for s in self.school.students 
                      if s.state == DiseaseState.INFECTIOUS]
        
        newly_exposed = [] 

        for infectious_student in infectious:
            contacts = self.simulate_contacts(infectious_student) 

            for contact in contacts:
                # determine transmission probability based on contact type 
                if (contact.grade == infectious_student.grade and 
                    contact.classroom == infectious_student.classroom):
                    beta = self.params.beta_within 
                else:
                    beta = self.params.beta_between 

                # attempt transmission
                if np.random.random() < beta:
                    newly_exposed.append(contact)

        # apply new exposures
        for student in newly_exposed:
            student.expose(self.current_time)

    def progression_step(self, dt: float = 1.0):
        """Advance disease progression for all infected individuals"""
        for student in self.school.students:
            if student.state in [DiseaseState.EXPOSED, DiseaseState.INFECTIOUS]:
                student.progress_disease(self.params, dt)

    def record_state(self):
        """Record current state of the epidemic"""
        counts = self.school.get_counts()
        classroom_counts = self.school.get_counts_by_classroom()

        record = {
            'time': self.current_time,
            'S': counts['S'],
            'E': counts['E'],
            'I': counts['I'],
            'R': counts['R'],
            'classroom_data': classroom_counts.copy()
        }
        self.history.append(record) 

    def run(self, n_days: int, dt: float = 1.0,
            record_interval: int = 1) -> List[Dict]:
        """Run the ABM simulation for specified duration.
        Parameters: 
        n_days: int. Number of days to simulate
        dt: float. Time step in days (default = 1.0)
        record_interval: int. Number of time steps between recordings 

        Returns:
        history: List[Dict]. Time series of epidemic states 
        """
        self.history = [] 
        self.current_time = 0.0
        self.record_state() # record initial state 

        n_steps = int(n_days / dt)

        for step in range(n_steps):
            # transmission dynamics
            self.transmission_step()

            # disease progression
            self.progression_step(dt)

            # update time
            self.current_time += dt 

            # record state periodically 
            if (step + 1) % record_interval == 0:
                self.record_state() 
        return self.history 
    
    def get_final_attack_rate(self) -> float:
        """Calculate final attack rate (fraction of susceptibles who got infected).
        Returns:
        attack_rate: float. Fraction of initially susceptible population that was infected
        """
        initial_susceptible = self.history[0]['S']
        final_susceptible = self.history[-1]['S']

        if initial_susceptible == 0:
            return 0.0 
        
        return (initial_susceptible - final_susceptible) / initial_susceptible 
    
    def get_epidemic_peak(self) -> Tuple[float, int]:
        """Find the epidemic peak.
        Returns:
        peak_time: float. Day of maximum infectious prevalence
        peak_infectious: int. Maximum number of infectious individuals
        """
        infectious_counts = [record['I'] for record in self.history]
        peak_infectious = max(infectious_counts)
        peak_idx = infectious_counts.index(peak_infectious)
        peak_time = self.history[peak_idx]['time']

        return peak_time, peak_infectious 
    
    
        