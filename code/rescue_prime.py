from algebra import *
from univariate import *
from multivariate import *

class RescuePrime:
    def __init__( self ):
        self.p = 407 * (1 << 119) + 1
        self.field = Field(self.p)
        self.m = 2
        self.rate = 1
        self.capacity = 1
        self.N = 27
        self.alpha = 3
        self.alphainv = 180331931428153586757283157844700080811
        self.MDS = [[FieldElement(v, self.field) for v in [270497897142230380135924736767050121214, 4]],
                    [FieldElement(v, self.field) for v in [270497897142230380135924736767050121205, 13]]]
        self.MDSinv = [[FieldElement(v, self.field) for v in [210387253332845851216830350818816760948, 60110643809384528919094385948233360270]],
                       [FieldElement(v, self.field) for v in [90165965714076793378641578922350040407, 180331931428153586757283157844700080811]]]
        self.round_constants = [FieldElement(v, self.field) for v in [174420698556543096520990950387834928928,
                                        109797589356993153279775383318666383471,
                                        228209559001143551442223248324541026000,
                                        268065703411175077628483247596226793933,
                                        250145786294793103303712876509736552288,
                                        154077925986488943960463842753819802236,
                                        204351119916823989032262966063401835731,
                                        57645879694647124999765652767459586992,
                                        102595110702094480597072290517349480965,
                                        8547439040206095323896524760274454544,
                                        50572190394727023982626065566525285390,
                                        87212354645973284136664042673979287772,
                                        64194686442324278631544434661927384193,
                                        23568247650578792137833165499572533289,
                                        264007385962234849237916966106429729444,
                                        227358300354534643391164539784212796168,
                                        179708233992972292788270914486717436725,
                                        102544935062767739638603684272741145148,
                                        65916940568893052493361867756647855734,
                                        144640159807528060664543800548526463356,
                                        58854991566939066418297427463486407598,
                                        144030533171309201969715569323510469388,
                                        264508722432906572066373216583268225708,
                                        22822825100935314666408731317941213728,
                                        33847779135505989201180138242500409760,
                                        146019284593100673590036640208621384175,
                                        51518045467620803302456472369449375741,
                                        73980612169525564135758195254813968438,
                                        31385101081646507577789564023348734881,
                                        270440021758749482599657914695597186347,
                                        185230877992845332344172234234093900282,
                                        210581925261995303483700331833844461519,
                                        233206235520000865382510460029939548462,
                                        178264060478215643105832556466392228683,
                                        69838834175855952450551936238929375468,
                                        75130152423898813192534713014890860884,
                                        59548275327570508231574439445023390415,
                                        43940979610564284967906719248029560342,
                                        95698099945510403318638730212513975543,
                                        77477281413246683919638580088082585351,
                                        206782304337497407273753387483545866988,
                                        141354674678885463410629926929791411677,
                                        19199940390616847185791261689448703536,
                                        177613618019817222931832611307175416361,
                                        267907751104005095811361156810067173120,
                                        33296937002574626161968730356414562829,
                                        63869971087730263431297345514089710163,
                                        200481282361858638356211874793723910968,
                                        69328322389827264175963301685224506573,
                                        239701591437699235962505536113880102063,
                                        17960711445525398132996203513667829940,
                                        219475635972825920849300179026969104558,
                                        230038611061931950901316413728344422823,
                                        149446814906994196814403811767389273580,
                                        25535582028106779796087284957910475912,
                                        93289417880348777872263904150910422367,
                                        4779480286211196984451238384230810357,
                                        208762241641328369347598009494500117007,
                                        34228805619823025763071411313049761059,
                                        158261639460060679368122984607245246072,
                                        65048656051037025727800046057154042857,
                                        134082885477766198947293095565706395050,
                                        23967684755547703714152865513907888630,
                                        8509910504689758897218307536423349149,
                                        232305018091414643115319608123377855094,
                                        170072389454430682177687789261779760420,
                                        62135161769871915508973643543011377095,
                                        15206455074148527786017895403501783555,
                                        201789266626211748844060539344508876901,
                                        179184798347291033565902633932801007181,
                                        9615415305648972863990712807943643216,
                                        95833504353120759807903032286346974132,
                                        181975981662825791627439958531194157276,
                                        267590267548392311337348990085222348350,
                                        49899900194200760923895805362651210299,
                                        89154519171560176870922732825690870368,
                                        265649728290587561988835145059696796797,
                                        140583850659111280842212115981043548773,
                                        266613908274746297875734026718148328473,
                                        236645120614796645424209995934912005038,
                                        265994065390091692951198742962775551587,
                                        59082836245981276360468435361137847418,
                                        26520064393601763202002257967586372271,
                                        108781692876845940775123575518154991932,
                                        138658034947980464912436420092172339656,
                                        45127926643030464660360100330441456786,
                                        210648707238405606524318597107528368459,
                                        42375307814689058540930810881506327698,
                                        237653383836912953043082350232373669114,
                                        236638771475482562810484106048928039069,
                                        168366677297979943348866069441526047857,
                                        195301262267610361172900534545341678525,
                                        2123819604855435621395010720102555908,
                                        96986567016099155020743003059932893278,
                                        248057324456138589201107100302767574618,
                                        198550227406618432920989444844179399959,
                                        177812676254201468976352471992022853250,
                                        211374136170376198628213577084029234846,
                                        105785712445518775732830634260671010540,
                                        122179368175793934687780753063673096166,
                                        126848216361173160497844444214866193172,
                                        22264167580742653700039698161547403113,
                                        234275908658634858929918842923795514466,
                                        189409811294589697028796856023159619258,
                                        75017033107075630953974011872571911999,
                                        144945344860351075586575129489570116296,
                                        261991152616933455169437121254310265934,
                                        18450316039330448878816627264054416127]]

    def hash( self, input_element ):
        # absorb
        state = [input_element] + [self.field.zero()] * (self.m - 1)

        # permutation
        for r in range(self.N):
            
            # forward half-round
            # S-box
            for i in range(self.m):
                state[i] = state[i]^self.alpha
            # matrix
            temp = [self.field.zero() for i in range(self.m)]
            for i in range(self.m):
                for j in range(self.m):
                    temp[i] = temp[i] + self.MDS[i][j] * state[j]
            # constants
            state = [temp[i] + self.round_constants[2*r*self.m+i] for i in range(self.m)]

            # backward half-round
            # S-box
            for i in range(self.m):
                state[i] = state[i]^self.alphainv
            # matrix
            temp = [self.field.zero() for i in range(self.m)]
            for i in range(self.m):
                for j in range(self.m):
                    temp[i] = temp[i] + self.MDS[i][j] * state[j]
            # constants
            state = [temp[i] + self.round_constants[2*r*self.m+self.m+i] for i in range(self.m)]

        # squeeze
        return state[0]

    def trace( self, input_element ):
        trace = []

        # absorb
        state = [input_element] + [self.field.zero()] * (self.m - 1)

        # explicit copy to record state into trace
        trace += [[s for s in state]]

        # permutation
        for r in range(self.N):
            
            # forward half-round
            # S-box
            for i in range(self.m):
                state[i] = state[i]^self.alpha
            # matrix
            temp = [self.field.zero() for i in range(self.m)]
            for i in range(self.m):
                for j in range(self.m):
                    temp[i] = temp[i] + self.MDS[i][j] * state[j]
            # constants
            state = [temp[i] + self.round_constants[2*r*self.m+i] for i in range(self.m)]

            # backward half-round
            # S-box
            for i in range(self.m):
                state[i] = state[i]^self.alphainv
            # matrix
            temp = [self.field.zero() for i in range(self.m)]
            for i in range(self.m):
                for j in range(self.m):
                    temp[i] = temp[i] + self.MDS[i][j] * state[j]
            # constants
            state = [temp[i] + self.round_constants[2*r*self.m+self.m+i] for i in range(self.m)]
            
            # record state at this point, with explicit copy
            trace += [[s for s in state]]

        # squeeze
        # output = state[0]

        return trace

    def boundary_constraints( self, output_element ):
        constraints = []

        # at start, capacity is zero
        constraints += [(0, 1, self.field.zero())]

        # at end, rate part is the given output element
        constraints += [(self.N, 0, output_element)]

        return constraints

    def round_constants_polynomials( self, omicron ):
        first_step_constants = []
        for i in range(self.m):
            domain = [omicron^r for r in range(0, self.N)]
            values = [self.round_constants[2*r*self.m+i] for r in range(0, self.N)]
            univariate = Polynomial.interpolate_domain(domain, values)
            multivariate = MPolynomial.lift(univariate, 0)
            first_step_constants += [multivariate]
        second_step_constants = []
        for i in range(self.m):
            domain = [omicron^r for r in range(0, self.N)]
            values = [self.field.zero()] * self.N
            #for r in range(self.N):
            #    print("len(round_constants):", len(self.round_constants), " but grabbing index:", 2*r*self.m+self.m+i, "for r=", r, "for m=", self.m, "for i=", i)
            #    values[r] = self.round_constants[2*r*self.m + self.m + i]
            values = [self.round_constants[2*r*self.m+self.m+i] for r in range(self.N)]
            univariate = Polynomial.interpolate_domain(domain, values)
            multivariate = MPolynomial.lift(univariate, 0)
            second_step_constants += [multivariate]

        return first_step_constants, second_step_constants

    def transition_constraints( self, omicron ):
        # get polynomials that interpolate through the round constants
        first_step_constants, second_step_constants = self.round_constants_polynomials(omicron)

        # arithmetize one round of Rescue-Prime
        variables = MPolynomial.variables(1 + 2*self.m, self.field)
        cycle_index = variables[0]
        previous_state = variables[1:(1+self.m)]
        next_state = variables[(1+self.m):(1+2*self.m)]
        air = []
        for i in range(self.m):
            # compute left hand side symbolically
            # lhs = sum(MPolynomial.constant(self.MDS[i][k]) * (previous_state[k]^self.alpha) for k in range(self.m)) + first_step_constants[i]
            lhs = MPolynomial.constant(self.field.zero())
            for k in range(self.m):
                lhs = lhs + MPolynomial.constant(self.MDS[i][k]) * (previous_state[k]^self.alpha)
            lhs = lhs + first_step_constants[i]

            # compute right hand side symbolically
            # rhs = sum(MPolynomial.constant(self.MDSinv[i][k]) * (next_state[k] - second_step_constants[k]) for k in range(self.m))^self.alpha
            rhs = MPolynomial.constant(self.field.zero())
            for k in range(self.m):
                rhs = rhs + MPolynomial.constant(self.MDSinv[i][k]) * (next_state[k] - second_step_constants[k])
            rhs = rhs^self.alpha

            # equate left and right hand sides
            air += [lhs-rhs]

        return air

    def randomizer_freedom( self, omicron, num_randomizers ):
        domain = [omicron^i for i in range(self.N, self.N+num_randomizers)]
        zerofier = Polynomial.zerofier_domain(domain)
        multivariate_zerofier = MPolynomial.lift(zerofier, 0)
        return multivariate_zerofier

