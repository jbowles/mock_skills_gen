package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"

	"github.com/icrowley/fake"
)

const (
	dSize = 20_000
)

func main() {
	rand.Seed(42)
	//rand.Seed(time.Now().UTC().UnixNano())
	// debug()

	randFile, err := os.Create("random_student_metrics.csv")
	if err != nil {
		log.Fatal(err)
	}
	GenerateRandomRaw(dSize, randFile)

	biasFile, err := os.Create("bias_student_metrics.csv")
	if err != nil {
		log.Fatal(err)
	}
	GenerateBiasRaw(dSize, biasFile)

}

func GenerateRandomRaw(n int, f *os.File) {
	defer f.Close()
	w := csv.NewWriter(f)
	err := w.Write(csvHeaders())
	if err != nil {
		log.Fatal(err)
	}
	for i := 0; i <= n; i++ {
		s := Student{}
		err = w.Write(s.buildRand().csvTransform())
		if err != nil {
			log.Fatal(err)
		}
	}

	w.Flush()
	if err = w.Error(); err != nil {
		log.Fatal(err)
	}
}
func GenerateBiasRaw(n int, f *os.File) {
	defer f.Close()
	w := csv.NewWriter(f)
	err := w.Write(csvHeaders())
	if err != nil {
		log.Fatal(err)
	}
	for i := 0; i <= n; i++ {
		s := Student{}
		err = w.Write(s.buildBias().csvTransform())
		if err != nil {
			log.Fatal(err)
		}
	}

	w.Flush()
	if err = w.Error(); err != nil {
		log.Fatal(err)
	}
}

type Gender int
type Target int

const (
	smUpper   = 100
	targetMax = 5
)
const (
	// NotReported Gender = -1
	Male Gender = iota
	Female
	NonBinary
)
const (
	Stem      Target = (targetMax - iota) //5
	Arts                                  //4
	Education                             //3
	GovBiz                                //2
	Medicine                              //1
)

var (
	targets = []Target{Stem, Arts, Education, GovBiz, Medicine}
)

type PersonData struct {
	FirstName   string
	LastName    string
	Age         int
	GenderLabel Gender
}
type GeneralMetrics struct {
	Spatial        int
	Temporal       int
	Verbal         int
	Organizational int
	Social         int
}
type StudentMetrics struct {
	General            GeneralMetrics
	TechnicalWriting   float64
	DescriptiveWriting float64
	AnalyticWriting    float64
	Arithmetic         float64
	Algebra            float64
	Geometry           float64
}
type Student struct {
	Pii PersonData
	M   StudentMetrics
	T   Target
}

func RandIntBetween(min int, max int) int {
	return min + rand.Intn(max-min)
}
func RandIntUnder(n int) int {
	i := rand.Intn(n)
	if i == 0 {
		return rand.Intn(n)
	}
	return i
}
func RandFloatUnder(n int) float64 {
	res := float64(RandIntUnder(n))
	return res + (res * 0.0034)
}
func RandGender(withNonBin bool) Gender {
	i := RandIntUnder(100)
	if withNonBin {
		if i == 33 {
			return NonBinary
		}
	}
	if i%2 == 0 {
		return Male
	} else {
		return Female
	}
}
func RandAge(min, max int) int {
	i := RandIntUnder(max)
	if i < min {
		return i + min
	}
	return i
}
func RandTarget() Target {
	return targets[RandIntUnder(targetMax)]
}

func (s *Student) buildRand() *Student {
	s.Pii = PersonData{
		FirstName:   fake.FirstName(),
		LastName:    fake.LastName(),
		Age:         RandAge(10, 30),
		GenderLabel: RandGender(false),
	}
	s.M = StudentMetrics{
		General: GeneralMetrics{
			Spatial:        RandIntUnder(smUpper),
			Temporal:       RandIntUnder(smUpper),
			Verbal:         RandIntUnder(smUpper),
			Organizational: RandIntUnder(smUpper),
			Social:         RandIntUnder(smUpper),
		},
		TechnicalWriting:   RandFloatUnder(smUpper),
		DescriptiveWriting: RandFloatUnder(smUpper),
		AnalyticWriting:    RandFloatUnder(smUpper),
		Arithmetic:         RandFloatUnder(smUpper),
		Algebra:            RandFloatUnder(smUpper),
		Geometry:           RandFloatUnder(smUpper),
	}
	s.T = RandTarget()
	return s
}

/*
>>> cb
Counter({3: 837, 2: 412, 1: 349, 4: 217, 5: 186})
>>> cr
Counter({1: 505, 2: 490, 4: 471, 3: 448, 5: 87})
*/
func (s *Student) buildBias() *Student {
	s.Pii = PersonData{
		FirstName:   fake.FirstName(),
		LastName:    fake.LastName(),
		Age:         RandAge(10, 30),
		GenderLabel: RandGender(false),
	}
	s.M, s.T = BiasStudent(rand.Int())
	return s
}

/*
	Stem        Target = (targetMax - iota) //5
	Arts                                    //4
	Education                               //3
	GovBiz                                  //2
	Medicine                                //1
*/
func BiasStudent(i int) (StudentMetrics, Target) {
	if i%11 == 0 {
		return mathStudent(), Stem
	}
	if i%7 == 0 {
		return litStudent(), Arts
	}
	if i%5 == 0 {
		return bioStudent(), Medicine
	}
	if i%2 == 0 {
		return bizStudent(), GovBiz
	}
	return eduStudent(), Education
}
func mathStudent() StudentMetrics {
	return StudentMetrics{
		General: GeneralMetrics{
			Spatial:        RandIntUnder(smUpper),
			Temporal:       RandIntUnder(smUpper),
			Verbal:         RandIntUnder(20),
			Organizational: RandIntUnder(20),
			Social:         RandIntUnder(20),
		},
		TechnicalWriting:   RandFloatUnder(30),
		DescriptiveWriting: RandFloatUnder(20),
		AnalyticWriting:    RandFloatUnder(50),
		Arithmetic:         RandFloatUnder(smUpper),
		Algebra:            RandFloatUnder(smUpper),
		Geometry:           RandFloatUnder(smUpper),
	}
}
func litStudent() StudentMetrics {
	return StudentMetrics{
		General: GeneralMetrics{
			Spatial:        RandIntUnder(40),
			Temporal:       RandIntUnder(30),
			Verbal:         RandIntUnder(smUpper),
			Organizational: RandIntUnder(smUpper),
			Social:         RandIntUnder(smUpper),
		},
		TechnicalWriting:   RandFloatUnder(90),
		DescriptiveWriting: RandFloatUnder(smUpper),
		AnalyticWriting:    RandFloatUnder(smUpper),
		Arithmetic:         RandFloatUnder(40),
		Algebra:            RandFloatUnder(20),
		Geometry:           RandFloatUnder(20),
	}
}
func bioStudent() StudentMetrics {
	return StudentMetrics{
		General: GeneralMetrics{
			Spatial:        RandIntUnder(smUpper),
			Temporal:       RandIntUnder(smUpper),
			Verbal:         RandIntUnder(30),
			Organizational: RandIntUnder(30),
			Social:         RandIntUnder(40),
		},
		TechnicalWriting:   RandFloatUnder(20),
		DescriptiveWriting: RandFloatUnder(50),
		AnalyticWriting:    RandFloatUnder(30),
		Arithmetic:         RandFloatUnder(smUpper),
		Algebra:            RandFloatUnder(smUpper),
		Geometry:           RandFloatUnder(smUpper),
	}
}
func bizStudent() StudentMetrics {
	return StudentMetrics{
		General: GeneralMetrics{
			Spatial:        RandIntUnder(20),
			Temporal:       RandIntUnder(20),
			Verbal:         RandIntUnder(smUpper),
			Organizational: RandIntUnder(smUpper),
			Social:         RandIntUnder(smUpper),
		},
		TechnicalWriting:   RandFloatUnder(30),
		DescriptiveWriting: RandFloatUnder(40),
		AnalyticWriting:    RandFloatUnder(20),
		Arithmetic:         RandFloatUnder(20),
		Algebra:            RandFloatUnder(30),
		Geometry:           RandFloatUnder(20),
	}
}
func eduStudent() StudentMetrics {
	return StudentMetrics{
		General: GeneralMetrics{
			Spatial:        RandIntUnder(30),
			Temporal:       RandIntUnder(20),
			Verbal:         RandIntUnder(smUpper),
			Organizational: RandIntUnder(smUpper),
			Social:         RandIntUnder(smUpper),
		},
		TechnicalWriting:   RandFloatUnder(smUpper),
		DescriptiveWriting: RandFloatUnder(smUpper),
		AnalyticWriting:    RandFloatUnder(smUpper),
		Arithmetic:         RandFloatUnder(60),
		Algebra:            RandFloatUnder(60),
		Geometry:           RandFloatUnder(60),
	}
}
func csvHeaders() []string {
	return []string{
		// "FirstName",
		// "LastName",
		"Age",
		"GenderLabel",
		"Spatial",
		"Temporal",
		"Verbal",
		"Organizational",
		"Social",
		"TechnicalWriting",
		"DescriptiveWriting",
		"AnalyticWriting",
		"Arithmetic",
		"Algebra",
		"Geometry",
		"Target",
	}
}

func (s *Student) csvTransform() []string {
	return []string{
		strconv.Itoa(s.Pii.Age),
		strconv.Itoa(int(s.Pii.GenderLabel)),
		strconv.Itoa(s.M.General.Spatial),
		strconv.Itoa(s.M.General.Temporal),
		strconv.Itoa(s.M.General.Verbal),
		strconv.Itoa(s.M.General.Organizational),
		strconv.Itoa(s.M.General.Social),
		// strconv.FormatFloat(s.M.TechnicalWriting, 'E', -1, 64),
		fmt.Sprintf("%v", s.M.TechnicalWriting),
		fmt.Sprintf("%v", s.M.DescriptiveWriting),
		fmt.Sprintf("%v", s.M.AnalyticWriting),
		fmt.Sprintf("%v", s.M.Arithmetic),
		fmt.Sprintf("%v", s.M.Algebra),
		fmt.Sprintf("%v", s.M.Geometry),
		strconv.Itoa(int(s.T)),
	}
}

/*
func debug() {
	fmt.Println("hello")
	for i := 0; i < 10; i++ {
		s := Student{}
		s.build()
		fmt.Println(s)
		fmt.Println(s.csvTransform())
		// fmt.Println(strconv.FormatFloat(s.M.TechnicalWriting, 'E', -1, 64))
		// fmt.Printf("%v", s.M.TechnicalWriting)
	}
}
*/

/*
type Proficiency int
const (
	None         Proficiency = 0
	Basic        Proficiency = 1
	Intermediate Proficiency = 2
	Advanced     Proficiency = 3
)
type Interest int
const (
	NoInterest    Interest = 0
	SomeInterest  Interest = 1
	GreatInterest Interest = 2
)
func (g Proficiency) String() string {
	switch g {
	case None:
		return "None"
	case Basic:
		return "Basic"
	case Intermediate:
		return "Intermediate"
	case Advanced:
		return "Advanced"
	default:
		return "ProficiencyNoMatch"
	}
}
func (g Interest) String() string {
	switch g {
	case NoInterest:
		return "NoInterest"
	case SomeInterest:
		return "SomeInterest"
	case GreatInterest:
		return "GreatInterest"
	default:
		return "InterestNoMatch"
	}
}

func (g Gender) String() string {
	switch g {
	case Male:
		return "Male"
	case Female:
		return "Female"
	case NonBinary:
		return "NonBinary"
	// case NotReported:
	// return "NotReported"
	default:
		return "GenderNoMatch"
	}
}
*/
