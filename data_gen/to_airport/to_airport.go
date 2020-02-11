package main

import (
	"encoding/csv"
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
	/*
		rand.Seed(42)

		biasFile, err := os.Create("to_airport_data.csv")
		if err != nil {
			log.Fatal(err)
		}
		GenerateToAirport(dSize, biasFile)
	*/
}

type ToAirport struct {
	Weekday   int
	TimeOfDay int
	Minutes   int
}

/*
// min returns smallest int between 2
func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

func split(input, dim int) []int {
	div := input / dim
	return fill(1, div)
}

func fill(min, max int) []int {
	res := []int{}
	for i := min; i <= max; i++ {
		res = append(res, i)
	}
	return res
}

// partition groups a slice of ints into chunks given partition size. last chunk will be remainder.
func partition(a []int, p int) [][]int {
	result := [][]int{}
	for i := 0; i < len(a); i += p {
		result = append(result, a[i:min(i+p, len(a))])
	}
	return result
}
func minMax2D(xinput [][]int) [][]int {
	result := [][]int{}

	for _, x := range xinput {
		smallest, biggest := x[0], x[0]
		for _, v := range x {
			if v > biggest {
				biggest = v
			}
			if v < smallest {
				smallest = v
			}
		}
		result = append(result, []int{smallest, biggest})
	}
	return result
}
*/

func GenerateToAirport(n int, f *os.File) {
	defer f.Close()
	w := csv.NewWriter(f)
	err := w.Write([]string{"weekday", "tod", "minutes"})
	if err != nil {
		log.Fatal(err)
	}
	for i := 0; i <= n; i++ {
		s := ToAirport{}
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

func RandIntBetween(min int, max int) int {
	return min + rand.Intn(max-min)
}

func RandFloatBetween(min, max, n int) float64 {
	res := float64(RandIntBetween(min, max))
	return res + (res * 0.0034)
}

func (s *ToAirport) buildRand() *ToAirport {
	s.Weekday = fake.WeekdayNum()
	s.Minutes = RandIntBetween(47, 103)
	s.TimeOfDay = RandIntBetween(6, 23)
	return s
}

func (s *ToAirport) csvTransform() []string {
	return []string{
		strconv.Itoa(s.Weekday),
		strconv.Itoa(s.TimeOfDay),
		strconv.Itoa(s.Minutes),
	}
}
